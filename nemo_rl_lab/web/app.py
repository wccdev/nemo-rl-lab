"""FastAPI 应用工厂。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nemo_rl_lab.web.auth import AuthGuard, UserStore, make_token
from nemo_rl_lab.web.config import WebSettings
from nemo_rl_lab.web.services.cluster import gpu_summary, run_status_map
from nemo_rl_lab.web.services.experiments import list_experiments
from nemo_rl_lab.web.services.ledger import enrich_runs_with_status, read_ledger
from nemo_rl_lab.web.services.ray_source import RayDataSource, resolve_exp_config


class LoginBody(BaseModel):
    username: str
    password: str


class CreateUserBody(BaseModel):
    username: str
    password: str
    role: str = "operator"


def create_app(settings: WebSettings) -> FastAPI:
    store = UserStore(settings.db_path)
    guard = AuthGuard(store, settings.jwt_secret, settings.no_auth)
    ray = RayDataSource(settings.ray_address, settings.repo_root, settings.cache_ttl)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield

    app = FastAPI(
        title="NeMo-RL Lab Console",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.no_auth else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- Auth ----------
    @app.post("/api/auth/login")
    def login(body: LoginBody):
        if settings.no_auth:
            return {"token": "local", "user": {"username": "local", "role": "admin"}}
        user = store.verify(body.username, body.password)
        if not user:
            raise HTTPException(401, "用户名或密码错误")
        return {"token": make_token(user, settings.jwt_secret, settings.jwt_hours), "user": user}

    @app.get("/api/auth/me")
    def me(user: dict = Depends(guard.current_user)):
        return user

    @app.post("/api/auth/setup")
    def setup_admin(body: CreateUserBody):
        if store.count_users() > 0:
            raise HTTPException(403, "已有用户，请用登录接口")
        store.create_user(body.username, body.password, role="admin")
        user = {"username": body.username, "role": "admin"}
        return {"ok": True, "token": make_token(user, settings.jwt_secret, settings.jwt_hours), "user": user}

    # ---------- Meta ----------
    @app.get("/api/auth/config")
    def auth_config():
        """公开：前端启动时判断是否本机免登模式。"""
        return {"no_auth": settings.no_auth}

    @app.get("/api/meta")
    def meta(user: dict = Depends(guard.current_user)):
        return {
            "ray_address": settings.ray_address,
            "no_auth": settings.no_auth,
            "user": user,
        }

    # ---------- Cluster ----------
    @app.get("/api/cluster/status")
    def cluster_status(user: dict = Depends(guard.current_user)):
        gpu = gpu_summary(settings.ray_address)
        jobs = [j for j in ray.list_jobs() if j["status"] in ("RUNNING", "PENDING")]
        return {"gpu": gpu, "active_jobs": jobs, "active_count": len(jobs)}

    # ---------- Runs ledger ----------
    @app.get("/api/runs")
    def runs(limit: int = Query(50, ge=1, le=500), user: dict = Depends(guard.current_user)):
        entries = read_ledger(settings.ledger_path)
        smap = run_status_map(settings.ray_address)
        rows = enrich_runs_with_status(entries, smap)[:limit]
        return {"runs": rows, "total": len(entries)}

    # ---------- Experiments ----------
    @app.get("/api/experiments")
    def experiments(user: dict = Depends(guard.current_user)):
        return {"experiments": list_experiments(settings.repo_root)}

    @app.get("/api/experiments/{name}/config")
    def experiment_config(name: str, user: dict = Depends(guard.current_user)):
        cfg = resolve_exp_config(settings.repo_root, name)
        if cfg is None:
            raise HTTPException(404, f"找不到实验 config: {name}")
        return {"name": name, "config": cfg}

    # ---------- Jobs ----------
    @app.get("/api/jobs")
    def jobs(user: dict = Depends(guard.current_user)):
        return ray.list_jobs()

    @app.get("/api/job")
    def job_overview(id: str = Query(...), user: dict = Depends(guard.current_user)):
        return ray.job_overview(id)

    @app.get("/api/diff")
    def config_diff(ids: str = Query(...), user: dict = Depends(guard.current_user)):
        id_list = [x.strip() for x in ids.split(",") if x.strip()]
        if not id_list:
            raise HTTPException(400, "ids 为空")
        return ray.config_diff(id_list)

    @app.get("/api/samples")
    def samples(
        id: str = Query(...),
        vidx: int = Query(...),
        offset: int = Query(0, ge=0),
        limit: int = Query(6, ge=1, le=50),
        user: dict = Depends(guard.current_user),
    ):
        return ray.samples_page(id, vidx, offset, limit)

    @app.post("/api/job/stop")
    def stop_job(id: str = Query(...), user: dict = Depends(guard.current_user)):
        return ray.stop_job(id)

    @app.post("/api/job/delete")
    def delete_job(id: str = Query(...), user: dict = Depends(guard.current_user)):
        return ray.delete_job(id)

    @app.get("/api/job/logs/stream")
    async def stream_logs(id: str = Query(...), user: dict = Depends(guard.current_user)):
        async def gen():
            last_len = 0
            while True:
                try:
                    text = ray.get_logs(id)
                    if len(text) > last_len:
                        yield text[last_len:]
                        last_len = len(text)
                except Exception as e:  # noqa: BLE001
                    yield f"\n[stream error] {e}\n"
                    break
                await asyncio.sleep(2)

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    # ---------- SPA static ----------
    static = settings.static_dir
    if static and static.is_dir():
        assets = static / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str, request: Request):
            if full_path.startswith("api/"):
                raise HTTPException(404)
            index = static / "index.html"
            if index.is_file():
                return FileResponse(index)
            raise HTTPException(404, "frontend not built")

    return app
