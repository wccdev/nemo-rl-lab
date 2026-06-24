"""中心化 Lab Server 的 CLI 侧：登录 / 凭据管理 / 命令门控。

仅用标准库（http.server / urllib / webbrowser）+ typer，不依赖 web extra，
保证未装 fastapi 的纯客户端也能 `lab login`。

本地状态：
  ~/.lab/config.json       {"server": "https://lab.company.com"}
  ~/.lab/credentials.json  {"<server>": {access_token, refresh_token, expires_at, user}}
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import typer

LAB_DIR = Path(os.environ.get("LAB_HOME") or (Path.home() / ".lab"))
CONFIG_PATH = LAB_DIR / "config.json"
CRED_PATH = LAB_DIR / "credentials.json"

# 需要登录才能用的集群类命令（server 模式下门控）；纯本地命令不在此列。
GATED_COMMANDS = {
    "submit", "export", "eval", "status", "logs",
    "job-list", "job-logs", "job-samples", "job-status",
    "job-stop", "job-delete", "job-clean", "job-cancel-all",
}


# ----------------------------- PKCE（stdlib）-----------------------------
def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


# ----------------------------- 本地配置/凭据 -----------------------------
def _read_json(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def current_server(explicit: Optional[str] = None) -> Optional[str]:
    """server 地址优先级：显式 > 环境 LAB_SERVER > config.json。"""
    s = explicit or os.environ.get("LAB_SERVER") or _read_json(CONFIG_PATH).get("server")
    return s.rstrip("/") if s else None


def is_server_mode() -> bool:
    return current_server() is not None


def _save_server(server: str) -> None:
    cfg = _read_json(CONFIG_PATH)
    cfg["server"] = server
    _write_json(CONFIG_PATH, cfg)


def _save_creds(server: str, creds: dict) -> None:
    all_creds = _read_json(CRED_PATH)
    all_creds[server] = creds
    _write_json(CRED_PATH, all_creds)


def _load_creds(server: str) -> Optional[dict]:
    return _read_json(CRED_PATH).get(server)


def _clear_creds(server: str) -> None:
    all_creds = _read_json(CRED_PATH)
    if server in all_creds:
        del all_creds[server]
        _write_json(CRED_PATH, all_creds)


# ----------------------------- HTTP（stdlib）-----------------------------
def _api(server: str, method: str, path: str, *, token: Optional[str] = None, body: Optional[dict] = None,
         timeout: float = 10.0) -> dict:
    url = f"{server}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


# ----------------------------- token 生命周期 -----------------------------
def _refresh(server: str, creds: dict) -> Optional[dict]:
    rt = creds.get("refresh_token")
    if not rt:
        return None
    try:
        resp = _api(server, "POST", "/api/auth/refresh", body={"refresh_token": rt})
    except urllib.error.HTTPError:
        return None
    creds["access_token"] = resp["access_token"]
    creds["expires_at"] = time.time() + resp.get("expires_in", 3600) - 60
    _save_creds(server, creds)
    return creds


def get_access_token(server: str, *, auto_refresh: bool = True) -> Optional[str]:
    """返回有效 access token；过期则用 refresh 续期；都不行返回 None。"""
    creds = _load_creds(server)
    if not creds:
        return None
    exp = creds.get("expires_at")
    if exp is None or time.time() < exp:
        return creds.get("access_token")
    if auto_refresh:
        refreshed = _refresh(server, creds)
        if refreshed:
            return refreshed["access_token"]
    return None


# ----------------------------- 代理提交（Phase B，客户端侧）-----------------------------
def _git_out(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_provenance(repo_root: Path, exp_rel: str) -> dict:
    """提交可追溯：git commit / dirty / config 指纹。"""
    commit = (_git_out(["rev-parse", "--short", "HEAD"], repo_root) or "unknown").strip()
    dirty = bool(_git_out(["status", "--porcelain"], repo_root).strip())
    cfg = repo_root / exp_rel / "config.yaml"
    if cfg.is_file():
        config_sha = hashlib.sha256(cfg.read_bytes()).hexdigest()[:12]
    else:
        config_sha = "none"
    return {"git_commit": commit, "git_dirty": dirty, "config_sha": config_sha}


def pack_working_dir(repo_root: Path) -> bytes:
    """打包 working-dir 为 tar.gz：用 git 跟踪 + 未忽略文件（精确遵循 .gitignore）。"""
    import io
    import tarfile

    listing = _git_out(["ls-files", "--cached", "--others", "--exclude-standard"], repo_root)
    files = [f for f in listing.splitlines() if f.strip()]
    if not files:
        raise typer.BadParameter("打包失败：未在 git 仓库内或没有可上传文件。")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in files:
            p = repo_root / rel
            if p.is_file():  # 跳过已删除/软链异常
                tar.add(p, arcname=rel)
    return buf.getvalue()


def _bearer_request(server: str, method: str, path: str, *, data: Optional[bytes] = None,
                    headers: Optional[dict] = None, timeout: float = 60.0):
    """带 token 的请求；返回 urlopen 的响应对象（调用方负责读取/关闭）。"""
    token = get_access_token(server)
    if not token:
        raise typer.BadParameter(f"未登录 {server}。先 `lab login`。")
    h = {"Authorization": f"Bearer {token}"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(f"{server}{path}", data=data, headers=h, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def submit_via_server(exp_rel: str, profile: Optional[str], repo_root: Path,
                      server: Optional[str] = None) -> dict:
    """server 模式提交：打包上传 + 服务端注入密钥后代理提交，返回 {job_id, run_id, ...}。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    meta = {"exp": exp_rel, "profile": profile or "", **git_provenance(repo_root, exp_rel)}
    blob = pack_working_dir(repo_root)
    headers = {"Content-Type": "application/gzip", "X-Lab-Meta": json.dumps(meta, ensure_ascii=False)}
    try:
        with _bearer_request(srv, "POST", "/api/jobs", data=blob, headers=headers, timeout=300.0) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        raise typer.BadParameter(f"提交失败 [{e.code}]：{detail}") from e


def submit_post_via_server(action: str, exp_rel: str, profile: Optional[str], flags: list[str],
                           repo_root: Path, server: Optional[str] = None) -> dict:
    """server 模式训练后闭环（export/eval）：打包上传 → 服务端代理提交 post_train.sh。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    meta = {"action": action, "exp": exp_rel, "profile": profile or "",
            "flags": flags, **git_provenance(repo_root, exp_rel)}
    blob = pack_working_dir(repo_root)
    headers = {"Content-Type": "application/gzip", "X-Lab-Meta": json.dumps(meta, ensure_ascii=False)}
    try:
        with _bearer_request(srv, "POST", "/api/post", data=blob, headers=headers, timeout=300.0) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        raise typer.BadParameter(f"{action} 提交失败 [{e.code}]：{detail}") from e


def usage_via_server(server: Optional[str] = None) -> dict:
    """server 模式：取本人配额 + 实时用量。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    with _bearer_request(srv, "GET", "/api/usage/mine") as r:
        return json.loads(r.read() or b"{}")


def list_my_jobs(server: Optional[str] = None, limit: int = 50) -> list[dict]:
    """server 模式：取本人作业（admin 取全部）。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    with _bearer_request(srv, "GET", f"/api/jobs/mine?limit={limit}") as r:
        return (json.loads(r.read() or b"{}")).get("jobs", [])


def job_control_via_server(action: str, job_id: str, server: Optional[str] = None) -> dict:
    """server 模式：停止 / 删除作业（经服务端，不直连 Ray）。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    path = f"/api/job/{action}?id={urllib.parse.quote(job_id)}"
    try:
        with _bearer_request(srv, "POST", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise typer.BadParameter(f"{action} 失败 [{e.code}]：{e.read().decode(errors='ignore')}") from e


def latest_job_via_server(server: Optional[str] = None) -> Optional[str]:
    srv = current_server(server)
    if not srv:
        return None
    try:
        with _bearer_request(srv, "GET", "/api/jobs/mine?limit=1") as r:
            jobs = (json.loads(r.read() or b"{}")).get("jobs", [])
    except urllib.error.HTTPError:
        return None
    return jobs[0].get("ray_submission_id") if jobs else None


def stream_logs_via_server(job_id: str, server: Optional[str] = None) -> None:
    """经服务端 SSE/流式接口跟随作业日志（客户端不直连 Ray）。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    path = f"/api/job/logs/stream?id={urllib.parse.quote(job_id)}"
    try:
        with _bearer_request(srv, "GET", path, timeout=None) as r:
            for chunk in r:
                sys.stdout.write(chunk.decode(errors="ignore"))
                sys.stdout.flush()
    except KeyboardInterrupt:
        typer.echo("\n(已停止跟随)")
    except urllib.error.HTTPError as e:
        raise typer.BadParameter(f"取日志失败 [{e.code}]") from e


def job_overview_via_server(job_id: str, server: Optional[str] = None) -> dict:
    """取作业概览（含 validations 列表），用于定位验证轮次。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    path = f"/api/job?id={urllib.parse.quote(job_id)}"
    try:
        with _bearer_request(srv, "GET", path) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise typer.BadParameter(f"取作业概览失败 [{e.code}]：{e.read().decode(errors='ignore')}") from e


def samples_via_server(job_id: str, vidx: int, offset: int = 0, limit: int = 6,
                       server: Optional[str] = None) -> dict:
    """取某次验证的多轮对话样本（分页）。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    q = urllib.parse.urlencode({"id": job_id, "vidx": vidx, "offset": offset, "limit": limit})
    try:
        with _bearer_request(srv, "GET", f"/api/samples?{q}") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise typer.BadParameter(f"取样本失败 [{e.code}]：{e.read().decode(errors='ignore')}") from e


def cluster_status_via_server(server: Optional[str] = None) -> Optional[dict]:
    """取集群 GPU 概览 + 活跃作业；Ray/服务不可达时返回 None（status 预检用）。"""
    srv = current_server(server)
    if not srv:
        return None
    try:
        with _bearer_request(srv, "GET", "/api/cluster/status") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError:
        return None


def batch_via_server(action: str, server: Optional[str] = None) -> dict:
    """批量作业控制：cancel-all（停止全部在跑）/ clean（清理终态 Ray 记录），经服务端。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。")
    try:
        with _bearer_request(srv, "POST", f"/api/jobs/{action}") as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise typer.BadParameter(f"{action} 失败 [{e.code}]：{e.read().decode(errors='ignore')}") from e


# ----------------------------- 回环登录流 -----------------------------
class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        qs = urllib.parse.parse_qs(parsed.query)
        type(self).result = {
            "code": (qs.get("code") or [None])[0],
            "state": (qs.get("state") or [None])[0],
            "error": (qs.get("error") or [None])[0],
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            "<html><body style='font-family:sans-serif;text-align:center;margin-top:80px'>"
            "<h2>✓ 登录成功</h2><p>已完成 CLI 授权，可关闭此页面回到终端。</p>"
            "</body></html>".encode()
        )

    def log_message(self, *a):  # 静默
        pass


def _browser_login(server: str, timeout: float = 180.0) -> dict:
    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(16)
    httpd = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    port = httpd.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    _CallbackHandler.result = {}

    q = urllib.parse.urlencode({"redirect_uri": redirect_uri, "state": state, "challenge": challenge})
    auth_url = f"{server}/cli/authorize?{q}"
    typer.echo(f"正在打开浏览器完成登录：{auth_url}")
    webbrowser.open(auth_url)

    deadline = time.time() + timeout

    def _serve():
        while not _CallbackHandler.result and time.time() < deadline:
            httpd.handle_request()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    t.join(timeout)
    httpd.server_close()

    res = _CallbackHandler.result
    if not res:
        raise typer.BadParameter("登录超时未收到回调，请重试。")
    if res.get("error"):
        raise typer.BadParameter(f"登录失败：{res['error']}")
    if res.get("state") != state:
        raise typer.BadParameter("state 校验失败（疑似 CSRF），已中止。")

    resp = _api(
        server, "POST", "/api/cli/token",
        body={"code": res["code"], "verifier": verifier, "redirect_uri": redirect_uri},
    )
    return {
        "access_token": resp["access_token"],
        "refresh_token": resp.get("refresh_token"),
        "expires_at": time.time() + resp.get("expires_in", 3600) - 60,
        "user": resp.get("user"),
    }


# ----------------------------- 命令门控 -----------------------------
def gate(command: str) -> None:
    """集群类命令执行前调用：未接入中心化服务则报错引导登录；已接入但未登录则自动开浏览器认证。"""
    server = current_server()
    if not server:
        raise typer.BadParameter(
            "未接入中心化服务。先 `lab login --server https://lab.company.com` 接入后再操作集群命令。"
        )
    if command not in GATED_COMMANDS:
        return
    token = get_access_token(server)
    if token:
        return
    typer.secho(f"未登录 {server}，正在打开浏览器完成认证…", fg=typer.colors.YELLOW)
    creds = _browser_login(server)
    _save_creds(server, creds)
    u = (creds.get("user") or {}).get("username", "?")
    typer.secho(f"✓ 已登录为 {u}", fg=typer.colors.GREEN)


# ----------------------------- Typer 命令 -----------------------------
def login(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="Lab 服务地址，如 https://lab.company.com"),
    token: Optional[str] = typer.Option(None, "--token", help="非交互登录：直接用服务令牌（CI 用）"),
) -> None:
    """登录中心化 Lab 服务（未登录命令会自动跳转浏览器）。"""
    srv = current_server(server)
    if not srv:
        raise typer.BadParameter("未指定服务地址。用 `lab login --server https://lab.company.com`。")
    _save_server(srv)
    if token:
        creds = {"access_token": token, "refresh_token": None, "expires_at": None, "user": None}
        try:
            who = _api(srv, "GET", "/api/whoami", token=token)
            creds["user"] = who.get("user")
        except urllib.error.HTTPError as e:
            raise typer.BadParameter(f"服务令牌无效：{e}") from e
        _save_creds(srv, creds)
    else:
        creds = _browser_login(srv)
        _save_creds(srv, creds)
    u = (creds.get("user") or {}).get("username", "?")
    typer.secho(f"✓ 已登录 {srv}（用户 {u}）", fg=typer.colors.GREEN)


def logout(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定服务地址（默认当前）"),
) -> None:
    """登出：吊销 refresh 并清除本地凭据。"""
    srv = current_server(server)
    if not srv:
        typer.echo("未登录任何服务。")
        return
    creds = _load_creds(srv)
    if creds and creds.get("refresh_token"):
        try:
            _api(srv, "POST", "/api/auth/logout", body={"refresh_token": creds["refresh_token"]})
        except urllib.error.URLError:
            pass
    _clear_creds(srv)
    typer.secho(f"✓ 已登出 {srv}", fg=typer.colors.GREEN)


def whoami(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定服务地址（默认当前）"),
) -> None:
    """显示当前登录身份 / 角色 / 配额。"""
    srv = current_server(server)
    if not srv:
        typer.echo("未接入中心化服务。用 `lab login --server https://lab.company.com` 接入。")
        return
    token = get_access_token(srv)
    if not token:
        typer.secho(f"未登录 {srv}。运行 `lab login` 登录。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    try:
        who = _api(srv, "GET", "/api/whoami", token=token)
    except urllib.error.HTTPError as e:
        typer.secho(f"查询失败：{e}", fg=typer.colors.RED)
        raise typer.Exit(1) from e
    user = who.get("user") or {}
    quota = who.get("quota") or {}
    typer.echo(f"服务：{srv}")
    typer.echo(f"用户：{user.get('username')}  角色：{user.get('role')}")
    typer.echo(
        "配额："
        f"GPU≤{quota.get('max_concurrent_gpus')} "
        f"作业≤{quota.get('max_concurrent_jobs')} "
        f"日GPU时={quota.get('daily_gpu_hours')}"
        + ("（默认/未单独分配）" if quota.get("default") else "")
    )


def quota(
    server: Optional[str] = typer.Option(None, "--server", "-s", help="指定服务地址（默认当前）"),
) -> None:
    """查看配额与实时用量（用量在 Phase C 接入）。"""
    srv = current_server(server)
    if not srv:
        typer.echo("未接入中心化服务。用 `lab login --server https://lab.company.com` 接入。")
        return
    token = get_access_token(srv)
    if not token:
        typer.secho(f"未登录 {srv}。运行 `lab login` 登录。", fg=typer.colors.YELLOW)
        raise typer.Exit(1)
    data = _api(srv, "GET", "/api/quota", token=token)
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


# ----------------------------- lab admin（管理员）-----------------------------
def _admin_call(method: str, path: str, *, body: Optional[dict] = None) -> dict:
    srv = current_server()
    if not srv:
        raise typer.BadParameter("未配置 Lab 服务。先 `lab login --server ...`。")
    token = get_access_token(srv)
    if not token:
        raise typer.BadParameter(f"未登录 {srv}。先 `lab login`。")
    try:
        return _api(srv, method, path, token=token, body=body)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        raise typer.BadParameter(f"请求失败 [{e.code}]：{detail}") from e


admin_app = typer.Typer(
    no_args_is_help=True,
    help="管理员：管理本地账号与算力配额（需以 admin 身份登录中心化 Lab 服务）。",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@admin_app.command("users", help="列出所有用户")
def admin_users() -> None:
    data = _admin_call("GET", "/api/admin/users")
    for u in data.get("users", []):
        flag = " [disabled]" if u.get("disabled") else ""
        typer.echo(f"{u['id']:>3}  {u['username']:<20} {u['role']:<10} {u.get('auth_source','')}{flag}")


@admin_app.command("user-add", help="新建本地账号")
def admin_user_add(
    username: str = typer.Argument(...),
    password: str = typer.Option(..., "--password", "-p", prompt=True, hide_input=True),
    role: str = typer.Option("operator", "--role", help="admin | operator | viewer"),
    email: Optional[str] = typer.Option(None, "--email"),
) -> None:
    u = _admin_call("POST", "/api/admin/users", body={"username": username, "password": password, "role": role, "email": email})
    typer.secho(f"✓ 已创建 {u['username']}（{u['role']}）", fg=typer.colors.GREEN)


@admin_app.command("set-role", help="修改用户角色")
def admin_set_role(username: str = typer.Argument(...), role: str = typer.Argument(...)) -> None:
    u = _admin_call("PATCH", f"/api/admin/users/{username}/role?role={urllib.parse.quote(role)}")
    typer.secho(f"✓ {u['username']} → {u['role']}", fg=typer.colors.GREEN)


@admin_app.command("disable", help="停用/启用用户（--on 停用，--off 启用）")
def admin_disable(
    username: str = typer.Argument(...),
    disabled: bool = typer.Option(True, "--on/--off", help="--on 停用，--off 启用"),
) -> None:
    u = _admin_call("PATCH", f"/api/admin/users/{username}/disabled?disabled={str(disabled).lower()}")
    typer.secho(f"✓ {u['username']} disabled={u['disabled']}", fg=typer.colors.GREEN)


@admin_app.command("set-quota", help="设置用户算力配额")
def admin_set_quota(
    username: str = typer.Argument(...),
    gpus: int = typer.Option(8, "--gpus", help="并发 GPU 上限"),
    jobs: int = typer.Option(4, "--jobs", help="并发作业上限"),
    daily_gpu_hours: int = typer.Option(0, "--daily-gpu-hours", help="每日 GPU-时（0=不限）"),
    profiles: Optional[str] = typer.Option(None, "--profiles", help="允许的 profile（逗号分隔，空=全部）"),
    priority: int = typer.Option(0, "--priority", help="排队优先级"),
) -> None:
    allowed = [p.strip() for p in profiles.split(",") if p.strip()] if profiles else []
    q = _admin_call("POST", "/api/admin/quotas", body={
        "username": username, "max_concurrent_gpus": gpus, "max_concurrent_jobs": jobs,
        "daily_gpu_hours": daily_gpu_hours, "allowed_profiles": allowed, "priority": priority,
    })
    typer.secho(f"✓ 已设置 {username} 配额：{json.dumps(q, ensure_ascii=False)}", fg=typer.colors.GREEN)
