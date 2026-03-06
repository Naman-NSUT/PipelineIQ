"""
🧟 Frankenstein Dataset Builder
================================
Takes the original Kaggle CI/CD dataset (45K rows, random features) and injects:
  1. 30-50 realistic error messages per failure category (with randomized params)
  2. Correlated tabular features (cpu, memory, retry, durations, booleans, stage)
  3. ~10-15% noise so the dataset isn't trivially separable

Usage:
    python build_frankenstein.py
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# 1. ERROR MESSAGE POOLS (30-50 per category, with templates)
# ═══════════════════════════════════════════════════════════

def _rand_ip():
    return f"{random.randint(10,192)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

def _rand_port():
    return random.choice([22, 80, 443, 3000, 3306, 5432, 6379, 8080, 8443, 9090, 27017])

def _rand_version():
    return f"{random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,99)}"

def _rand_class():
    names = ["UserService", "PaymentController", "AuthManager", "DatabasePool", "CacheHandler",
             "EventProcessor", "FileUploader", "NotificationSender", "OrderService", "ApiGateway"]
    return random.choice(names)

def _rand_file():
    exts = [".py", ".js", ".ts", ".java", ".go", ".rb", ".rs"]
    names = ["app", "main", "server", "handler", "controller", "service", "utils", "config", "worker", "index"]
    return f"src/{random.choice(names)}{random.choice(exts)}"

def _rand_pkg():
    return random.choice(["numpy", "pandas", "flask", "django", "requests", "boto3", "sqlalchemy",
                           "celery", "redis", "pydantic", "fastapi", "torch", "scipy", "pillow"])

def _rand_npm():
    return random.choice(["react", "express", "lodash", "axios", "webpack", "next", "prisma",
                           "typescript", "jest", "eslint", "vite", "tailwindcss", "zod"])

def _rand_cve():
    return f"CVE-{random.randint(2019,2025)}-{random.randint(10000,99999)}"


ERROR_TEMPLATES = {
    'Build Failure': [
        "BUILD FAILED: Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.8.1:compile",
        f"make: *** [Makefile:{random.randint(10,200)}: all] Error 2",
        "error TS2304: Cannot find name 'React'. Check your tsconfig.json",
        "fatal error: Python.h: No such file or directory. compilation terminated.",
        lambda: f"FAILURE: Build failed with an exception at {_rand_file()}:{random.randint(1,500)}",
        "error: linker command failed with exit code 1 (use -v to see invocation)",
        "error[E0308]: mismatched types — expected `String`, found `&str`",
        lambda: f"Build step 'Execute shell' marked build as failure at line {random.randint(50,300)}",
        "FAILURE: Build failed — Could not resolve all dependencies for configuration ':app:releaseCompileClasspath'",
        "error: cannot find symbol class BuildConfig in package com.example.app",
        lambda: f"go build: {_rand_file()}: undefined: {_rand_class()}.Initialize",
        "CMake Error: The source directory does not appear to contain CMakeLists.txt",
        "error: script returned exit code 1. Build step 'Invoke Gradle script' failed.",
        lambda: f"SyntaxError: Unexpected token '{{' at {_rand_file()}:{random.randint(1,200)}",
        "cargo build failed: could not compile `serde_derive` due to 3 previous errors",
        lambda: f"javac: error: cannot access {_rand_class()}: class file not found",
        "error CS1022: Type or namespace definition, or end-of-file expected",
        "BUILD FAILED: npm ERR! Failed at the build script. npm ERR! This is probably not a problem with npm.",
        lambda: f"./gradlew: line {random.randint(1,100)}: exec: java: not found",
        "Error: Module build failed (from ./node_modules/babel-loader/lib/index.js)",
        lambda: f"gcc: error: {_rand_file()}: No such file or directory",
        "rustc --edition=2021 src/main.rs: error: aborting due to 5 previous errors",
        "FAILURE: Build failed with an exception. What went wrong: Execution failed for task ':compileJava'.",
        lambda: f"ld: symbol(s) not found for architecture x86_64 in {_rand_file()}",
        "error: Swift compiler error: value of type 'Optional<String>' has no member 'count'",
        "Build FAILED. 0 Warning(s). 12 Error(s). Time Elapsed 00:00:14.32",
        lambda: f"tsc: error TS2339: Property 'data' does not exist on type 'Response' at {_rand_file()}:{random.randint(1,300)}",
        "clang: error: no input files. compilation terminated.",
        lambda: f"Error: Cannot find module '{_rand_npm()}' — did you forget to install it?",
        "fatal: Could not read from remote repository during submodule update",
    ],

    'Test Failure': [
        lambda: f"AssertionError: Expected status [200] but got [{random.choice([400,403,404,500,502,503])}]",
        lambda: f"FAILED tests/test_api.py::test_{random.choice(['login','signup','payment','search','delete'])} - assert False",
        lambda: f"java.lang.NullPointerException at com.example.service.{_rand_class()}Test.test{random.choice(['Find','Create','Update','Delete'])}",
        "1 passing (2s) 3 failing. Uncaught TypeError: Cannot read properties of undefined (reading 'map')",
        lambda: f"FAIL: {_rand_class()}Test > should return correct result FAILED. Expected: 42, Actual: null",
        lambda: f"pytest: {random.randint(1,5)} passed, {random.randint(1,8)} failed, {random.randint(0,3)} errors in {random.uniform(1,30):.1f}s",
        "test session starts — platform linux — Python 3.9.7. FAILURES: test_integration_flow",
        lambda: f"Error: expect(received).toBe(expected) // Object.is equality. Expected: {random.randint(1,100)}, Received: {random.randint(1,100)}",
        "cucumber.runtime.CucumberException: No step definitions found for 'Given the user is logged in'",
        lambda: f"Traceback: File \"{_rand_file()}\", line {random.randint(10,200)}, in test_handler. AssertionError",
        "XFAIL: test marked as expected failure but it passed unexpectedly",
        "JUnit: Tests run: 42, Failures: 3, Errors: 1, Skipped: 0",
        lambda: f"RSpec: {random.randint(30,100)} examples, {random.randint(1,10)} failures — {_rand_class()} spec failed",
        "Selenium: TimeoutException: Element not clickable at point (350, 220). Other element would receive the click",
        lambda: f"mocha: {random.randint(1,5)} failing — TypeError: {_rand_class()}.{random.choice(['process','validate','transform'])} is not a function",
        "E2E test failed: Login flow — expected URL to include '/dashboard' but got '/error'",
        "AssertionError: Response body mismatch. diff: - expected + actual",
        lambda: f"go test ./... FAIL {_rand_class()}_test.go:{random.randint(10,100)}: unexpected EOF",
        "PHPUnit: FAILURES! Tests: 28, Assertions: 95, Failures: 4.",
        lambda: f"jest: Test suite failed to run — SyntaxError: {_rand_file()}: Unexpected identifier",
        "Coverage threshold not met: Branches 62% < 80% minimum. Build failed.",
        "FAIL: TestCalculateDiscount (0.00s) — got 0.15, want 0.20",
        lambda: f"nose2: ERROR: test_{random.choice(['create','read','update','delete'])}_{random.choice(['user','order','product'])} ({_rand_class()})",
        "playwright: page.click: Timeout 30000ms exceeded. selector='button[data-testid=\"submit\"]'",
        lambda: f"AssertionError: len(result) == {random.randint(0,2)}, expected {random.randint(3,10)}",
        "cypress: CypressError: Timed out retrying after 4000ms: Expected to find element: '[data-cy=login-form]'",
        lambda: f"unittest: FAIL: test_{random.choice(['api','model','view','serializer'])} ({_rand_class()}Test). {random.randint(1,3)} failures",
        "snapshot test failed: Received value does not match stored snapshot 1",
        lambda: f"LoadError: cannot load such file -- spec/{_rand_class().lower()}_spec",
        "FAILED: mutation testing — 15 mutants survived out of 42 (64% killed, threshold: 80%)",
    ],

    'Network Error': [
        lambda: f"botocore.exceptions.EndpointConnectionError: Could not connect to endpoint URL https://{_rand_ip()}:{_rand_port()}",
        lambda: f"Connection timed out: connect to {_rand_ip()} port {_rand_port()}: Connection timed out after 30s",
        lambda: f"curl: (28) Failed to connect to {_rand_ip()} port {_rand_port()}: Connection timed out",
        "java.net.SocketTimeoutException: Read timed out after 60000ms",
        lambda: f"ETIMEDOUT: connect ETIMEDOUT {_rand_ip()}:{_rand_port()} — network is unreachable",
        lambda: f"ssl.SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate for {_rand_ip()}",
        "requests.exceptions.ConnectionError: HTTPSConnectionPool: Max retries exceeded with url: /api/v1/health",
        lambda: f"DNS resolution failed: Could not resolve hostname registry.{random.choice(['npmjs.org','pypi.org','docker.io','maven.org'])}",
        lambda: f"ECONNREFUSED: connect ECONNREFUSED {_rand_ip()}:{_rand_port()} — connection actively refused",
        "net/http: request canceled while waiting for connection (Client.Timeout exceeded: 30s)",
        lambda: f"psycopg2.OperationalError: could not connect to server: Connection refused. Is the server running on {_rand_ip()} port {_rand_port()}?",
        "redis.exceptions.ConnectionError: Error 111 connecting to redis:6379. Connection refused.",
        lambda: f"HTTP 502 Bad Gateway: upstream server at {_rand_ip()}:{_rand_port()} did not respond",
        "urllib3.exceptions.MaxRetryError: HTTPSConnectionPool: Max retries exceeded (Caused by NewConnectionError)",
        lambda: f"gRPC: UNAVAILABLE: failed to connect to all addresses; last error: connect to {_rand_ip()}:{_rand_port()}: Connection refused",
        lambda: f"WebSocket connection to 'wss://{_rand_ip()}:{_rand_port()}' failed: Connection closed before handshake",
        "fetch failed: TypeError: Failed to fetch — net::ERR_CONNECTION_RESET",
        lambda: f"S3: An error occurred (RequestTimeout): Your socket connection to the server was not read within the timeout period",
        "io.netty.channel.ConnectTimeoutException: connection timed out after 10000 ms",
        lambda: f"ssh: connect to host {_rand_ip()} port 22: Connection timed out. fatal: Could not read from remote repository.",
        "AMQP ConnectionError: Connection to RabbitMQ lost. Retrying in 5s...",
        lambda: f"MongoDB: MongoNetworkTimeoutError: connection timed out to {_rand_ip()}:{_rand_port()}",
        "Error: getaddrinfo ENOTFOUND api.internal.service — DNS lookup failed",
        lambda: f"HTTP 504 Gateway Timeout from upstream {_rand_ip()} after 60s",
        "DockerError: Error while pulling image: Get https://registry-1.docker.io/v2/: net/http: TLS handshake timeout",
        lambda: f"grpcio: StatusCode.DEADLINE_EXCEEDED. Deadline exceeded after {random.uniform(10,60):.1f}s",
        "ElasticSearch: ConnectionError: ConnectionError(<urllib3.connection.HTTPConnection>: Connection refused)",
        lambda: f"kafka.errors.NoBrokersAvailable: NoBrokersAvailable — all brokers at {_rand_ip()}:{_rand_port()} are down",
        "SocketException: Connection reset by peer during artifact download",
        "java.net.UnknownHostException: packages.internal.corp: Name or service not known",
    ],

    'Deployment Failure': [
        "Error: Container exited with code 137 (OOMKilled) during deployment",
        lambda: f"Deployment failed: Health check endpoint /health returned HTTP {random.choice([500,502,503])} after 120s",
        "ERROR: ECS service deployment failed — tasks failed to start. Essential container exited.",
        lambda: f"kubectl: error: deployment '{_rand_class().lower()}-service' exceeded its progress deadline",
        "Helm: Error: UPGRADE FAILED: timed out waiting for the condition. Release rolled back.",
        "AWS CodeDeploy: ApplicationStop lifecycle event failed with exit code 1",
        "ArgoCD: sync failed: rpc error: code = Unknown desc = Manifest generation error",
        lambda: f"Terraform: Error applying plan: 1 error(s) occurred: aws_instance.{_rand_class().lower()}: Error launching source instance",
        "Docker Compose: ERROR: for web — Container exited with code 1. cannot start service web",
        "Deployment rolled back: New version failed readiness probe 3/3 times",
        lambda: f"Ansible: fatal: [{_rand_ip()}]: FAILED! => {{\"msg\": \"Service {_rand_class().lower()} failed to start\"}}",
        "Blue-green deployment aborted: target group health check failed for new environment",
        "Vercel: Error: Build failed — Function size limit exceeded (50MB max). Deployment rolled back.",
        "Istio: upstream connect error or disconnect/reset before headers. reset reason: connection failure",
        "Cloud Run: The user-provided container failed to start and listen on port 8080 within 240 seconds",
        lambda: f"GKE: Pod {_rand_class().lower()}-{random.randint(1000,9999)} CrashLoopBackOff — restarted {random.randint(5,20)} times",
        "Spinnaker: Pipeline execution failed at Deploy stage: Unexpected task failure",
        "Error: Canary deployment aborted — error rate 12.5% exceeds threshold 5.0%",
        "Pulumi: error: update failed: resource aws:ecs:Service failed: 0/2 tasks are running",
        lambda: f"Nomad: Allocation failed: failed to place allocation for job '{_rand_class().lower()}-api'",
        "Flyway: Migration failed — migration checksum mismatch for V003__add_users_table.sql",
        "CircleCI: Deployment to production failed: approval step timed out after 60 minutes",
        lambda: f"Lambda: Runtime.ImportModuleError: Unable to import module '{_rand_class().lower()}_handler': No module named '{_rand_pkg()}'",
        "Azure DevOps: Release deployment failed — agent disconnected during file copy",
        lambda: f"PM2: Process {_rand_class().lower()} errored — restart count: {random.randint(10,50)}. Stopped.",
        "CloudFormation: Stack update failed — UPDATE_ROLLBACK_COMPLETE. Reason: Resource creation cancelled",
        "Jenkins: Post-deployment smoke test failed. Rolling back to previous stable version.",
        "Render: Deploy failed — Build exited with non-zero code. Check build logs for details.",
        lambda: f"OpenShift: DeploymentConfig '{_rand_class().lower()}' failed — LatestDeploymentFailed",
        "Railway: Deployment crashed — process exited with SIGKILL. Out of memory.",
    ],

    'Dependency Error': [
        lambda: f"ModuleNotFoundError: No module named '{_rand_pkg()}'",
        lambda: f"npm ERR! code E404 — 404 Not Found: GET https://registry.npmjs.org/{_rand_npm()}@{_rand_version()}",
        lambda: f"go: {random.choice(['github.com','gitlab.com'])}/{_rand_class().lower()}/{_rand_class().lower()}@v{_rand_version()}: reading go.mod: 404 Not Found",
        "pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org'): Read timed out",
        lambda: f"Could not find artifact com.example:{_rand_class().lower()}:{_rand_version()} in central (https://repo1.maven.org/maven2)",
        lambda: f"npm WARN deprecated {_rand_npm()}@{_rand_version()}: This package is no longer maintained",
        lambda: f"CONFLICT: {_rand_pkg()}>={random.randint(1,3)}.0 requires numpy>={random.randint(1,2)}.{random.randint(0,20)} but you have numpy=={random.randint(1,2)}.{random.randint(0,20)}",
        "gem install: ERROR: Failed to build native extension for nokogiri",
        lambda: f"cargo: error: failed to select a version for `{random.choice(['serde','tokio','reqwest','clap'])}` — required by {_rand_class().lower()}",
        "Error: unable to resolve dependency tree. npm ERR! Fix the upstream dependency conflict.",
        lambda: f"ImportError: cannot import name '{random.choice(['BaseModel','Field','Optional'])}' from '{_rand_pkg()}'",
        "composer: Your requirements could not be resolved to an installable set of packages. Problem 1:",
        lambda: f"yarn: error: An unexpected error occurred: 'https://registry.yarnpkg.com/{_rand_npm()}': ETIMEDOUT",
        "pip: ERROR: No matching distribution found for tensorflow-gpu==2.12.0; python_version >= '3.12'",
        lambda: f"gradle: Could not resolve {_rand_class().lower()}:{_rand_version()}. Required by: project :app",
        "nuget: NU1102 — Unable to find package Microsoft.EntityFrameworkCore.SqlServer with version (>= 7.0.0)",
        lambda: f"poetry: SolverProblemError — {_rand_pkg()} requires python ^3.8 but your python version is 3.12.1",
        "bundler: Fetching source index from https://rubygems.org/ — Could not find gem 'rails (~> 7.0)' in rubygems repository",
        lambda: f"apt-get: E: Unable to locate package lib{_rand_pkg()}-dev",
        "vcpkg: Error: Failed to download dependency: SHA512 mismatch for downloaded file",
        lambda: f"mix deps.get: Failed to fetch dependency {_rand_class().lower()} from hex.pm: 404",
        "pipenv: ResolutionFailure — pipenv found conflicting versions: requests>=2.28 and requests<2.25",
        lambda: f"Error: peer dep missing: {_rand_npm()}@^{random.randint(1,5)}.0.0, required by {_rand_npm()}@{_rand_version()}",
        "CocoaPods: CDN Error — trunk URL couldn't be downloaded: https://cdn.cocoapods.org/",
        lambda: f"dotnet restore failed: NU1101 — Unable to find package {_rand_class()}.{random.choice(['Core','Client','Abstractions'])}",
        "conda: ResolvePackageNotFound: - cudatoolkit==11.8 [not found in current channels]",
        lambda: f"bower ECONFLICT: Unable to find suitable version for {_rand_npm()}: ^{_rand_version()} vs ~{_rand_version()}",
        "pnpm: ERR_PNPM_PEER_DEP_ISSUES — Unmet peer dependencies detected in lockfile",
        lambda: f"hex: Dependency resolution failed: {_rand_class().lower()} {_rand_version()} requires elixir ~> 1.12 but you have 1.11.4",
        "stack: Error: While constructing the build plan, the following exceptions were encountered: package not found in snapshot",
    ],

    'Configuration Error': [
        lambda: f"Error: Environment variable '{random.choice(['DATABASE_URL','API_KEY','SECRET_KEY','AWS_ACCESS_KEY_ID','REDIS_URL'])}' is not set",
        "yaml.scanner.ScannerError: while parsing a block mapping — found unexpected ':' in config.yml",
        lambda: f"ValidationError: Port {random.randint(1,65535)} is already in use. Check your configuration.",
        "Error: Invalid configuration — 'database.host' is required but was not provided",
        "toml: Error: expected value, found newline at line 42 in pyproject.toml",
        "docker-compose.yml: service 'web' references non-existent volume 'data_volume'",
        lambda: f"nginx: [emerg] unknown directive \"{random.choice(['proxy_pass','upstream','server_name'])}\" in /etc/nginx/nginx.conf:{random.randint(10,80)}",
        "Error: Configuration file 'application.properties' not found in classpath",
        "terraform plan: Error: Missing required argument — 'region' must be specified in provider 'aws'",
        lambda: f"JSON: Parse error at line {random.randint(1,50)}: Expected ',' or '}}' but found string in config.json",
        "Kubernetes: error validating data: unknown field 'replcias' in io.k8s.api.apps.v1.DeploymentSpec",
        "ESLint: Configuration for rule 'no-unused-vars' is invalid — Value 'warnning' should be one of: off, warn, error",
        lambda: f"Spring: Failed to bind properties under '{random.choice(['spring.datasource','server','logging'])}' to {_rand_class()}",
        "dotenv: Error: .env file not found at /app/.env. Running without environment file.",
        lambda: f"apache: AH00526: Syntax error on line {random.randint(10,200)} of /etc/apache2/apache2.conf",
        "Webpack: Invalid configuration object — 'output.filename' must be a string",
        "CircleCI: Unable to parse config file — Error: YAML syntax error at line 38: bad indentation of a mapping entry",
        lambda: f"Flask: Error: Could not locate a Flask application — set FLASK_APP={_rand_file()}",
        "Ansible: ERROR! 'rol' is not a valid attribute for a Play — did you mean 'role'?",
        "Error: Invalid connection string format. Expected 'postgresql://user:pass@host:port/db'",
        lambda: f"Systemd: Failed to start {_rand_class().lower()}.service: Unit configuration has fatal error",
        "Redis: Bad directive or wrong number of arguments in redis.conf at line 42",
        "Gunicorn: Error: Config file 'gunicorn.conf.py' not found. Using default settings.",
        "pm2: Error: Script not found: /app/dist/server.js — check ecosystem.config.js",
        lambda: f"CMake: Error at CMakeLists.txt:{random.randint(1,50)}: Unknown CMake command '{random.choice(['add_libary','set_traget','inclue'])}'",
        "SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in chain",
        "Vite: Error: Preprocessing config — invalid 'build.rollupOptions.external' value",
        lambda: f"supervisord: Error in config file /etc/supervisor/conf.d/{_rand_class().lower()}.conf: bad entry 'comand'",
        "Error: tsconfig.json — Compiler option 'moduleResolution' requires a value of 'node', 'node16', or 'bundler'",
        "Caddy: Error adapting config: Caddyfile:12 — unknown directive 'reverse_proxxy'",
    ],

    'Resource Exhaustion': [
        "FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed — JavaScript heap out of memory",
        "java.lang.OutOfMemoryError: Java heap space — consider increasing -Xmx",
        "Killed: 9 (OOMKiller invoked by kernel — process used 98% of available memory)",
        "Error: ENOSPC: no space left on device, write '/tmp/build-artifacts.tar.gz'",
        lambda: f"Container killed: memory usage {random.randint(900,1900)}Mi exceeds limit {random.choice([512, 1024])}Mi",
        "ThreadPoolExecutor: Cannot submit task — pool is full (max_workers=16, queue_size=1000)",
        lambda: f"PostgreSQL: FATAL: too many connections for role 'app' — limit is {random.choice([25,50,100])}, current is {random.choice([26,51,101])}",
        "OSError: [Errno 24] Too many open files — ulimit is 1024",
        lambda: f"Docker: no space left on device — /var/lib/docker is {random.randint(95,100)}% full",
        "GC overhead limit exceeded — garbage collection is taking >98% of CPU time",
        "MySQL: ERROR 1040 (HY000): Too many connections (max_connections = 151)",
        lambda: f"Process consumed {random.randint(28000,32000)}MB of memory. Terminated by cgroup OOM killer.",
        "Redis: OOM command not allowed when used memory > 'maxmemory'. maxmemory=256mb",
        lambda: f"Worker timeout: worker PID {random.randint(1000,9999)} exceeded time limit of {random.choice([30,60,120])}s — killed",
        "npm: ENOMEM — not enough memory, available: 128MB, required: 512MB for node_modules install",
        lambda: f"Disk usage critical: {random.randint(96,100)}% on /dev/sda1 ({random.randint(45,100)}GB / {random.choice([50, 100])}GB)",
        "Kubernetes: Pod evicted due to ephemeral storage usage exceeding limit (5Gi > 4Gi)",
        "Python: MemoryError: Unable to allocate 4.2 GiB for an array with shape (560000000,)",
        lambda: f"CUDA out of memory: Tried to allocate {random.randint(1,8)}GB. GPU 0 has {random.choice([4,6,8])}GB total capacity.",
        "Celery: WorkerLostError — Worker exited prematurely: signal 9 (SIGKILL) — likely OOM",
        "AWS Lambda: Runtime.ExitError — Task timed out after 900.00 seconds (max memory used: 3008 MB)",
        lambda: f"ElasticSearch: CircuitBreakingException: [parent] Data too large, data for [request] would be [{random.randint(80,99)}%] of JVM heap",
        "Linux: swap space exhausted. System is thrashing — consider adding more RAM or reducing memory usage.",
        "BuildKit: error: failed to solve — executor failed running: out of disk space for layer cache",
        lambda: f"Connection pool exhausted: All {random.choice([10,20,50])} connections are in use. Consider increasing pool size.",
        "Rust: thread 'main' panicked at 'memory allocation of 2147483648 bytes failed', alloc.rs:233:5",
        "Go: runtime: out of memory: cannot allocate — goroutine stack exceeds 1000000000-byte limit",
        "GitHub Actions: The runner has received a shutdown signal — job exceeded maximum execution time of 6h",
        lambda: f"RabbitMQ: connection.blocked — memory alarm triggered at {random.randint(90,99)}% usage",
        "Vercel: Error: Function execution exceeded 60s timeout. Consider optimizing or upgrading your plan.",
    ],

    'Permission Error': [
        lambda: f"PermissionError: [Errno 13] Permission denied: '{_rand_file()}'",
        "HTTP 403 Forbidden: You don't have permission to access this resource",
        "fatal: could not read Username for 'https://github.com': No such device or address",
        lambda: f"AWS: AccessDenied — User arn:aws:iam::{random.randint(100000000000,999999999999)}:user/ci-bot is not authorized to perform: s3:PutObject",
        "ssh: Permission denied (publickey). fatal: Could not read from remote repository.",
        "docker: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock",
        "kubectl: error: You must be logged in to the server (Unauthorized)",
        lambda: f"GCP: IAM permission '{random.choice(['compute.instances.create','storage.objects.get','container.pods.exec'])}' denied on resource",
        "npm ERR! Error: EACCES: permission denied, mkdir '/usr/local/lib/node_modules'",
        "git push: remote: Permission to user/repo.git denied to ci-bot. fatal: unable to access",
        lambda: f"Azure: AuthorizationFailed — The client '{random.choice(['ci-pipeline','deploy-bot','build-agent'])}' does not have authorization to perform action",
        "sudo: no tty present and no askpass program specified. Permission denied.",
        "pip: ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: '/usr/lib/python3/dist-packages'",
        "Vault: Error making API request: Code: 403. Errors: 1 error occurred: permission denied",
        lambda: f"Firebase: PERMISSION_DENIED: Missing or insufficient permissions for document '{_rand_class().lower()}/{random.randint(1,999)}'",
        "RBAC: Access denied — role 'developer' lacks permission 'deploy:production'",
        "ECR: CannotPullContainerError: AccessDeniedException: User is not authorized to perform ecr:GetAuthorizationToken",
        "chmod: changing permissions of '/opt/app/secrets.env': Operation not permitted",
        lambda: f"Artifactory: 401 Unauthorized — API key expired or invalid for repo '{random.choice(['maven-releases','npm-local','docker-prod'])}'",
        "Error: EPERM: operation not permitted, unlink '/var/lock/dpkg-lock'",
        "Kubernetes: pods is forbidden: User 'system:serviceaccount:ci:default' cannot list resource 'pods'",
        "AWS STS: ExpiredToken — The security token included in the request is expired",
        "GCS: AccessDenied — does not have storage.objects.create access to the Google Cloud Storage bucket",
        lambda: f"LDAP: Simple bind failed — invalid credentials for user cn=ci-bot,dc=example,dc=com",
        "GitHub API: 401 Bad credentials — personal access token has expired",
        "Docker Hub: denied: requested access to the resource is denied. unauthorized: authentication required",
        "OpenSSL: error:0A000418: routines::certificate key mismatch — SSL handshake failed",
        "Jenkins: Access denied. ci-agent is missing the Overall/Read permission",
        "Error: GITHUB_TOKEN does not have 'packages:write' scope. Cannot publish to GitHub Packages.",
        "Azure DevOps: VS402392 — Personal access token has expired. Please create a new one.",
    ],

    'Security Scan Failure': [
        lambda: f"VULNERABILITY FOUND: {_rand_cve()} (CRITICAL) detected in pom.xml — CVSS Score: {random.uniform(8.0,10.0):.1f}",
        "gitleaks: Secret detected: AWS Access Key ID found in config/deploy.yml",
        "SonarQube Quality Gate FAILED: Bugs > 0, Vulnerabilities > 0, Code Smells > 50",
        lambda: f"Dependency-Check: {random.randint(1,8)} HIGH severity vulnerabilities found in jackson-databind-{_rand_version()}.jar",
        lambda: f"Snyk: Found {random.randint(3,25)} vulnerabilities ({random.randint(1,5)} critical, {random.randint(1,10)} high) in package.json",
        "Trivy: CRITICAL vulnerability detected in base image node:14-alpine — upgrade to node:18-alpine",
        lambda: f"Bandit: {random.randint(2,10)} security issues found. High severity: {random.randint(1,5)} (B301, B602, B108)",
        "SAST: SQL Injection vulnerability detected at src/database/queries.py:42 — use parameterized queries",
        "Container scan: image contains root user — violates security policy SCR-003",
        lambda: f"npm audit: found {random.randint(5,50)} vulnerabilities ({random.randint(1,5)} critical). Run `npm audit fix` to resolve.",
        "OWASP ZAP: DAST scan failed — XSS vulnerability found at /api/search?q=<script>alert(1)</script>",
        "Safety: 3 vulnerabilities found in requirements.txt — urllib3<2.0.0a1 has CVE-2023-43804",
        "Checkmarx: High severity finding — Reflected Cross-Site Scripting at UserController.java:128",
        "tfsec: CRITICAL: Secret key is hard-coded in Terraform module — aws_secret_access_key detected",
        lambda: f"Grype: {random.randint(1,5)} Critical, {random.randint(2,10)} High vulnerabilities in container image {_rand_class().lower()}:latest",
        "license-checker: FAILED — Package 'crypto-js' uses license 'AGPL-3.0' which is not in the allowed list",
        "Clair: 2 fixable vulnerabilities found in layer sha256:a1b2c3... — libssl1.1 needs upgrade",
        "gosec: G101 (CWE-798): Potential hardcoded credentials at main.go:55 — variable 'dbPassword'",
        "DAST: Open redirect vulnerability found at /oauth/callback?redirect_uri=https://evil.com",
        lambda: f"Semgrep: {random.randint(3,15)} findings across {random.randint(2,8)} files — severity: {random.randint(1,3)} error, {random.randint(2,12)} warning",
        "Docker Bench: WARN 4.5 — Container running as root user. Use USER directive.",
        lambda: f"pip-audit: Found {random.randint(1,8)} known vulnerabilities in {random.randint(1,5)} packages in requirements.txt",
        "Secrets scan: GitHub token with repo scope found in .github/workflows/deploy.yml:18",
        "Veracode: Scan FAILED — CWE-89: Improper Neutralization of SQL Commands found in 3 locations",
        "Image signing verification failed: cosign signature not found for image digest sha256:abc123...",
        "SBOM: License compliance check failed — 4 packages with incompatible licenses detected",
        "Fortify: 2 Critical, 5 High priority issues — Path Manipulation detected in FileUploadService.java",
        "ScoutSuite: AWS Security audit FAILED — 12 findings: S3 buckets with public access, unencrypted EBS volumes",
        "CredScan: Pre-commit hook detected potential secrets — API key pattern found in .env.example",
        lambda: f"WhiteSource: Policy violation — {_rand_pkg()} {_rand_version()} has known exploit {_rand_cve()}",
    ],

    'Timeout': [
        lambda: f"Error: Job exceeded maximum execution time of {random.choice([10,15,30,60])} minutes. Cancelled.",
        lambda: f"TimeoutError: Process did not complete within {random.choice([300,600,900,1800,3600])}s deadline",
        "CircleCI: Too long with no output (exceeded 10m0s): context deadline exceeded",
        "GitHub Actions: The job running on runner 'ubuntu-latest' has exceeded the maximum execution time of 6 hours",
        lambda: f"Watchdog: Process {_rand_class().lower()} unresponsive for {random.choice([60,120,300])}s — terminated with SIGTERM",
        "Build timeout: Pipeline stage 'integration-tests' exceeded 45 minute limit",
        lambda: f"Deadlock detected: Thread pool exhausted after {random.choice([30,60,120])}s — all workers blocked",
        "Travis CI: The job exceeded the maximum time limit of 50 minutes",
        "AWS CodeBuild: Build timed out — phase BUILD exceeded time limit of 60 minutes",
        lambda: f"Kubernetes: Job 'batch-process' backoff limit reached — {random.randint(3,6)} failures in {random.choice([300,600])}s",
        "Jenkins: Build timed out (after 30 minutes). Sending interrupt signal to process.",
        "Pytest: Timeout — test_heavy_computation exceeded 120s limit (default: 60s)",
        lambda: f"Health check timeout: Service '{_rand_class().lower()}' did not respond within {random.choice([10,30,60])}s startup window",
        "Azure DevOps: The pipeline has been running for more than 60 minutes. Cancelling.",
        "GitLab CI: ERROR: Job failed: execution took longer than 1h0m0s seconds",
        lambda: f"Lock wait timeout exceeded: Transaction #{random.randint(1000,9999)} — try restarting transaction",
        "Celery: TimeLimitExceeded — Task hard time limit (1800s) reached. Worker terminated.",
        "Docker build: Step 8/12 timed out after 600s — likely a network issue during package install",
        lambda: f"gRPC: DeadlineExceeded — RPC to {_rand_class()}.{random.choice(['Process','Execute','Sync'])} did not complete in {random.choice([5,10,30])}s",
        "Terraform: Error: timeout while waiting for state to become 'running' (last state: 'pending', timeout: 10m0s)",
        "Buildkite: Command timed out after 20 minutes with no output",
        lambda: f"Queue processing timeout: Message in '{_rand_class().lower()}-queue' exceeded visibility timeout of {random.choice([30,60,300])}s",
        "Netlify: Deploy timed out — build did not complete within 30 minutes",
        "Maven: Build did not finish within the configured timeout of 600 seconds",
        lambda: f"Connection idle timeout: Database connection pool reclaimed connection after {random.choice([300,600,1800])}s idle",
        "Gradle: Daemon became idle and was stopped after 120s — build must be restarted",
        "ArgoCD: sync timed out — application did not reach healthy state within 300s",
        "Bazel: Build interrupted — exceeded 1800s wall-clock time limit",
        lambda: f"Prometheus: Scrape timeout for target {_rand_ip()}:{_rand_port()} — context deadline exceeded",
        "Firebase: Function execution took more than 540s — increase timeout or optimize function",
    ],
}


# ═══════════════════════════════════════════════════════════
# 2. TABULAR FEATURE CORRELATIONS
# ═══════════════════════════════════════════════════════════

FEATURE_PROFILES = {
    # ── GROUP 1: "Build-phase" — Build, Config, Dependency share similar profiles ──
    'Build Failure': {
        'cpu_usage_pct': (20, 70),
        'memory_usage_mb': (1000, 12000),
        'retry_count': (0, 2),
        'build_duration_sec': (100, 2000),
        'test_duration_sec': (0, 200),
        'deploy_duration_sec': (0, 60),
        'is_flaky_test': 0.12,
        'rollback_triggered': 0.15,
        'incident_created': 0.20,
        'failure_stage': 'build',
    },
    'Configuration Error': {
        'cpu_usage_pct': (15, 65),
        'memory_usage_mb': (500, 10000),
        'retry_count': (0, 2),
        'build_duration_sec': (30, 1500),
        'test_duration_sec': (0, 200),
        'deploy_duration_sec': (10, 400),
        'is_flaky_test': 0.10,
        'rollback_triggered': 0.20,
        'incident_created': 0.25,
        'failure_stage': 'build',
    },
    'Dependency Error': {
        'cpu_usage_pct': (15, 60),
        'memory_usage_mb': (500, 10000),
        'retry_count': (0, 3),
        'build_duration_sec': (30, 1500),
        'test_duration_sec': (0, 150),
        'deploy_duration_sec': (0, 60),
        'is_flaky_test': 0.10,
        'rollback_triggered': 0.12,
        'incident_created': 0.18,
        'failure_stage': 'build',
    },
    # ── GROUP 2: "Runtime" — Network, Timeout, Deployment IDENTICAL ranges ──
    'Network Error': {
        'cpu_usage_pct': (20, 75),
        'memory_usage_mb': (1000, 16000),
        'retry_count': (1, 4),
        'build_duration_sec': (50, 1200),
        'test_duration_sec': (10, 600),
        'deploy_duration_sec': (60, 1200),
        'is_flaky_test': 0.18,
        'rollback_triggered': 0.25,
        'incident_created': 0.35,
        'failure_stage': 'deploy',
    },
    'Timeout': {
        'cpu_usage_pct': (20, 75),
        'memory_usage_mb': (1000, 16000),
        'retry_count': (1, 3),
        'build_duration_sec': (100, 2000),
        'test_duration_sec': (60, 800),
        'deploy_duration_sec': (60, 1200),
        'is_flaky_test': 0.20,
        'rollback_triggered': 0.28,
        'incident_created': 0.38,
        'failure_stage': 'build',
    },
    'Deployment Failure': {
        'cpu_usage_pct': (25, 75),
        'memory_usage_mb': (1500, 16000),
        'retry_count': (0, 3),
        'build_duration_sec': (60, 1200),
        'test_duration_sec': (30, 500),
        'deploy_duration_sec': (100, 1500),
        'is_flaky_test': 0.12,
        'rollback_triggered': 0.35,
        'incident_created': 0.40,
        'failure_stage': 'deploy',
    },
    # ── GROUP 3: "Security/Access" — Permission and Security Scan share profiles ──
    'Permission Error': {
        'cpu_usage_pct': (10, 50),
        'memory_usage_mb': (256, 8000),
        'retry_count': (0, 2),
        'build_duration_sec': (10, 600),
        'test_duration_sec': (0, 200),
        'deploy_duration_sec': (10, 400),
        'is_flaky_test': 0.08,
        'rollback_triggered': 0.18,
        'incident_created': 0.30,
        'failure_stage': 'deploy',
    },
    'Security Scan Failure': {
        'cpu_usage_pct': (10, 50),
        'memory_usage_mb': (256, 8000),
        'retry_count': (0, 1),
        'build_duration_sec': (30, 600),
        'test_duration_sec': (20, 300),
        'deploy_duration_sec': (0, 100),
        'is_flaky_test': 0.08,
        'rollback_triggered': 0.15,
        'incident_created': 0.35,
        'failure_stage': 'test',
    },
    # ── GROUP 4: Test Failure — mid-range, overlaps with everyone ──
    'Test Failure': {
        'cpu_usage_pct': (20, 70),
        'memory_usage_mb': (1000, 12000),
        'retry_count': (0, 2),
        'build_duration_sec': (60, 1000),
        'test_duration_sec': (100, 1500),
        'deploy_duration_sec': (0, 60),
        'is_flaky_test': 0.35,
        'rollback_triggered': 0.12,
        'incident_created': 0.22,
        'failure_stage': 'test',
    },
    # ── GROUP 5: Resource Exhaustion — only class with distinctive signature ──
    'Resource Exhaustion': {
        'cpu_usage_pct': (70, 98),
        'memory_usage_mb': (16000, 32000),
        'retry_count': (0, 1),
        'build_duration_sec': (200, 2000),
        'test_duration_sec': (60, 800),
        'deploy_duration_sec': (60, 800),
        'is_flaky_test': 0.15,
        'rollback_triggered': 0.30,
        'incident_created': 0.45,
        'failure_stage': 'build',
    },
}

# Stage mapping — more ambiguous, less signal for XGBoost
STAGE_MAP = {
    'Build Failure':        {'build': 0.55, 'test': 0.25, 'deploy': 0.20},
    'Test Failure':         {'build': 0.15, 'test': 0.55, 'deploy': 0.30},
    'Network Error':        {'build': 0.25, 'test': 0.25, 'deploy': 0.50},
    'Deployment Failure':   {'build': 0.10, 'test': 0.15, 'deploy': 0.75},
    'Dependency Error':     {'build': 0.55, 'test': 0.20, 'deploy': 0.25},
    'Configuration Error':  {'build': 0.40, 'test': 0.20, 'deploy': 0.40},
    'Resource Exhaustion':  {'build': 0.35, 'test': 0.30, 'deploy': 0.35},
    'Permission Error':     {'build': 0.25, 'test': 0.20, 'deploy': 0.55},
    'Security Scan Failure':{'build': 0.30, 'test': 0.40, 'deploy': 0.30},
    'Timeout':              {'build': 0.33, 'test': 0.34, 'deploy': 0.33},
}

NOISE_RATE = 0.27  # 27% of rows get completely random noise in features



# ═══════════════════════════════════════════════════════════
# 3. BUILD THE DATASET
# ═══════════════════════════════════════════════════════════

def generate_error_message(failure_type):
    """Pick a random error message template and evaluate it if it's a lambda."""
    pool = ERROR_TEMPLATES[failure_type]
    template = random.choice(pool)
    if callable(template):
        return template()
    return template


def apply_tabular_features(row):
    """Override tabular features based on failure type profile."""
    ft = row['failure_type']
    profile = FEATURE_PROFILES[ft]

    # Apply noise: 12% chance of random values instead of correlated
    if random.random() < NOISE_RATE:
        return row  # Keep original random values

    # Numeric features (uniform random within range)
    row['cpu_usage_pct'] = round(np.random.uniform(*profile['cpu_usage_pct']), 1)
    row['memory_usage_mb'] = round(np.random.uniform(*profile['memory_usage_mb']), 1)
    row['retry_count'] = np.random.randint(profile['retry_count'][0], profile['retry_count'][1] + 1)
    row['build_duration_sec'] = round(np.random.uniform(*profile['build_duration_sec']), 1)
    row['test_duration_sec'] = round(np.random.uniform(*profile['test_duration_sec']), 1)
    row['deploy_duration_sec'] = round(np.random.uniform(*profile['deploy_duration_sec']), 1)

    # Boolean features (Bernoulli)
    row['is_flaky_test'] = random.random() < profile['is_flaky_test']
    row['rollback_triggered'] = random.random() < profile['rollback_triggered']
    row['incident_created'] = random.random() < profile['incident_created']

    # Failure stage (weighted random based on mapping)
    stage_probs = STAGE_MAP[ft]
    row['failure_stage'] = np.random.choice(
        list(stage_probs.keys()),
        p=list(stage_probs.values())
    )

    return row


def main():
    print("=" * 60)
    print("  🧟 FRANKENSTEIN DATASET BUILDER")
    print("=" * 60)

    # 1. Load original dataset
    print("\n📦 Loading original Kaggle dataset...")
    df = pd.read_csv('ci_cd_pipeline_failure_logs_dataset.csv')
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Failure types: {df['failure_type'].nunique()}")

    # 2. Verify all failure types have templates
    missing = set(df['failure_type'].unique()) - set(ERROR_TEMPLATES.keys())
    if missing:
        print(f"   ⚠️  Missing templates for: {missing}")
        return
    print(f"   ✅ Templates cover all {len(ERROR_TEMPLATES)} categories")

    # Print template counts
    for ft, templates in ERROR_TEMPLATES.items():
        print(f"      {ft}: {len(templates)} templates")

    # 3. Inject error messages
    print("\n🔧 Injecting realistic error messages...")
    df['error_message'] = df['failure_type'].apply(generate_error_message)

    # Check uniqueness
    unique_msgs = df['error_message'].nunique()
    print(f"   Unique error messages generated: {unique_msgs}")

    # 4. Fix tabular features
    print("\n🔧 Correlating tabular features with failure types...")
    df = df.apply(apply_tabular_features, axis=1)

    # 5. Quick sanity check — show correlations
    print("\n📊 Feature correlations with target (should be non-zero):")
    le = LabelEncoder()
    target_encoded = le.fit_transform(df['failure_type'])
    for col in ['cpu_usage_pct', 'memory_usage_mb', 'retry_count',
                'build_duration_sec', 'test_duration_sec', 'deploy_duration_sec']:
        corr = np.corrcoef(df[col].astype(float), target_encoded)[0, 1]
        signal = "✅" if abs(corr) > 0.05 else "⚠️"
        print(f"   {signal} {col}: {corr:.4f}")

    # 6. Save
    output_file = 'frankenstein_cicd_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file} ({len(df)} rows × {len(df.columns)} cols)")

    # 7. Summary
    print(f"\n{'=' * 60}")
    print(f"  ✅ FRANKENSTEIN DATASET READY!")
    print(f"  📁 File: {output_file}")
    print(f"  📊 Rows: {len(df)}")
    print(f"  📋 Columns: {len(df.columns)}")
    print(f"  🎯 Classes: {df['failure_type'].nunique()}")
    print(f"  💬 Unique messages: {unique_msgs}")
    print(f"  🔊 Noise rate: {NOISE_RATE*100:.0f}%")
    print(f"{'=' * 60}")
    print("\n🚀 Run your hybrid pipeline with:")
    print(f"   df = pd.read_csv('{output_file}')")


if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    main()
