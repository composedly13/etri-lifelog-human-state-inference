# ============================================================
# run_baseline.ps1
# ETRI 휴먼이해 AI 논문경진대회 — LightGBM Baseline 실행 스크립트
# 인코딩: UTF-8
#
# 사용법:
#   powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1 -Config configs/base.yaml
# ============================================================

param(
    [string]$Config = "configs/base.yaml"
)

$ErrorActionPreference = "Stop"

# ── 컬러 출력 헬퍼 ──────────────────────────────────────────
function Write-Header { param([string]$msg)
    Write-Host "`n$('='*55)" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "$('='*55)" -ForegroundColor Cyan
}
function Write-Step { param([string]$msg)
    Write-Host "`n[STEP] $msg" -ForegroundColor Yellow
}
function Write-Ok { param([string]$msg)
    Write-Host "  [OK]  $msg" -ForegroundColor Green
}
function Write-Warn { param([string]$msg)
    Write-Host "  [!!]  $msg" -ForegroundColor Red
}

# ── 시작 ────────────────────────────────────────────────────
Write-Header "ETRI 휴먼이해 AI 논문경진대회 — Baseline 실행"

# ── 1. 현재 위치 확인 ────────────────────────────────────────
Write-Step "현재 디렉토리 확인"
$currentDir = Get-Location
Write-Host "  현재 위치: $currentDir"

# 프로젝트 루트 확인 (configs/base.yaml 존재 여부로 판단)
if (-not (Test-Path "configs/base.yaml")) {
    Write-Warn "configs/base.yaml 파일을 찾을 수 없습니다."
    Write-Warn "프로젝트 루트 디렉토리에서 실행하세요."
    Write-Host ""
    Write-Host "  올바른 실행 방법:"
    Write-Host "    cd D:\etri-lifelog-human-state-inference"
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1"
    exit 1
}
Write-Ok "configs/base.yaml 확인됨"

# ── 2. Python 버전 확인 ──────────────────────────────────────
Write-Step "Python 버전 확인"
try {
    $pythonVersion = python --version 2>&1
    Write-Ok "Python: $pythonVersion"

    # Python 3.10 이상 확인
    $versionStr = $pythonVersion -replace "Python ", ""
    $major, $minor = $versionStr.Split('.')[0,1]
    if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
        Write-Warn "Python 3.10 이상을 권장합니다. 현재: $versionStr"
    }
} catch {
    Write-Warn "Python을 찾을 수 없습니다."
    Write-Host "  Python 3.10+ 설치 후 다시 시도하세요: https://www.python.org/"
    exit 1
}

# ── 3. 데이터 파일 확인 ──────────────────────────────────────
Write-Step "데이터 파일 확인"

$dataFiles = @(
    @{Path="data/raw/train.csv";              Name="Train"},
    @{Path="data/raw/test.csv";               Name="Test"},
    @{Path="data/raw/sample_submission.csv";  Name="Sample Submission"}
)

$dataMissing = $false
foreach ($file in $dataFiles) {
    if (Test-Path $file.Path) {
        $size = (Get-Item $file.Path).Length / 1KB
        Write-Ok "$($file.Name): $($file.Path) ($([math]::Round($size, 1)) KB)"
    } else {
        Write-Warn "$($file.Name): $($file.Path) — 파일 없음!"
        $dataMissing = $true
    }
}

if ($dataMissing) {
    Write-Host ""
    Write-Warn "데이터 파일이 없습니다."
    Write-Host "  1. 대회 데이터를 다음 위치에 배치하세요:"
    Write-Host "       data/raw/train.csv"
    Write-Host "       data/raw/test.csv"
    Write-Host "       data/raw/sample_submission.csv"
    Write-Host ""
    Write-Host "  2. 또는 configs/base.yaml에서 경로를 변경하세요:"
    Write-Host "       paths.data_dir: `"/data`""
    Write-Host ""
    Write-Host "  3. 데이터 없이 코드 구조만 확인하려면:"
    Write-Host "       python src/config.py"
    exit 1
}

# ── 4. 패키지 확인 ──────────────────────────────────────────
Write-Step "필수 패키지 확인"

$packages = @("lightgbm", "sklearn", "pandas", "numpy", "yaml")
$missingPkgs = @()

foreach ($pkg in $packages) {
    $check = python -c "import $pkg; print('ok')" 2>&1
    if ($check -eq "ok") {
        Write-Ok "$pkg"
    } else {
        Write-Warn "$pkg — 미설치"
        $missingPkgs += $pkg
    }
}

if ($missingPkgs.Count -gt 0) {
    Write-Host ""
    Write-Warn "일부 패키지가 설치되지 않았습니다."
    Write-Host "  다음 명령어로 설치하세요:"
    Write-Host "    pip install -r requirements.txt"
    Write-Host ""
    $response = Read-Host "  설치 후 계속하시겠습니까? (y/n)"
    if ($response -ne 'y') {
        exit 1
    }
}

# ── 5. 출력 디렉토리 준비 ─────────────────────────────────────
Write-Step "출력 디렉토리 확인"
$outDirs = @("outputs/logs", "outputs/models", "outputs/submissions", "outputs/figures")
foreach ($d in $outDirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    Write-Ok $d
}

# ── 6. Config 내용 출력 ──────────────────────────────────────
Write-Step "Config 확인"
python src/config.py

# ── 7. 학습 실행 ────────────────────────────────────────────
Write-Header "LightGBM Baseline 학습 시작"
Write-Host "  Config: $Config"
Write-Host "  시작시각: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

$startTime = Get-Date

try {
    python src/train_lgbm.py --config $Config
    $exitCode = $LASTEXITCODE
} catch {
    Write-Warn "학습 중 오류 발생: $_"
    exit 1
}

$elapsed = (Get-Date) - $startTime

# ── 8. 결과 확인 ────────────────────────────────────────────
Write-Header "실행 완료"
Write-Host "  소요 시간: $($elapsed.ToString('mm\분 ss\초'))"
Write-Host "  종료 코드: $exitCode"
Write-Host ""

if ($exitCode -eq 0) {
    Write-Ok "학습 성공!"
    Write-Host ""
    Write-Host "  결과 파일 위치:"

    # 최신 submission 파일
    $submissions = Get-ChildItem "outputs/submissions/*.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($submissions) {
        Write-Host "  [제출 파일]"
        $submissions | Select-Object -First 3 | ForEach-Object {
            Write-Host "    $($_.FullName)"
        }
    }

    # 최신 로그 파일
    $logs = Get-ChildItem "outputs/logs/*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($logs) {
        Write-Host "  [로그 파일]"
        $logs | Select-Object -First 2 | ForEach-Object {
            Write-Host "    $($_.FullName)"
        }
    }

    # Summary JSON
    $summaries = Get-ChildItem "outputs/logs/*_summary.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    if ($summaries) {
        Write-Host "  [실험 요약]"
        $summaries | Select-Object -First 1 | ForEach-Object {
            Write-Host "    $($_.FullName)"
            # JSON 내용 일부 출력
            try {
                $json = Get-Content $_.FullName -Raw | ConvertFrom-Json
                Write-Host "    실험명: $($json.experiment)"
            } catch {}
        }
    }

    Write-Host ""
    Write-Host "  다음 단계:"
    Write-Host "    1. outputs/submissions/ 에서 제출 파일 확인"
    Write-Host "    2. docs/experiment_log.md 에 OOF 스코어 기록"
    Write-Host "    3. 오늘 남은 제출 횟수 확인 (하루 최대 3회)"

} else {
    Write-Warn "학습 실패 (exit code: $exitCode)"
    Write-Host ""
    Write-Host "  디버깅 방법:"
    Write-Host "    python src/config.py          # 설정 확인"
    Write-Host "    python src/load_data.py        # 데이터 로드 테스트"
    Write-Host "    python src/train_lgbm.py --config $Config  # 직접 실행 후 오류 확인"
}

Write-Host ""
Write-Host "$('='*55)" -ForegroundColor Cyan
