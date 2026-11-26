param(
    [string]$OutDir = "runs/local-preflight-pilot"
)

$py = "C:\Users\Hoang\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $py)) {
    throw "Python 3.11 CUDA runtime not found at $py"
}

Write-Host "1/4 Running unit tests..."
& $py -m unittest discover -s tests -v
if ($LASTEXITCODE) { exit $LASTEXITCODE }
node --test mvp/replay.test.js
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host "2/4 Running CUDA smoke fixture train on local Quadro T2000..."
if (Test-Path "runs/local-smoke") { Remove-Item "runs/local-smoke" -Recurse -Force }
New-Item -ItemType Directory -Force -Path ".cache/local_smoke_raw", ".cache/local_smoke_prep" | Out-Null
& $py -m omnishield fixture --raw .cache/local_smoke_raw --prepared .cache/local_smoke_prep
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $py -m omnishield train --raw .cache/local_smoke_raw --prepared .cache/local_smoke_prep --config configs/train/smoke.json --allow-synthetic --out runs/local-smoke --workers 0 --amp off --device cuda
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host "3/4 Running real-data 8-frame preflight train on local Quadro T2000..."
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
& $py -m omnishield train `
    --raw data/raw/phenikaa-mini-pilot `
    --prepared data/prepared/phenikaa-mini-pilot `
    --config configs/train/mvp_preflight.json `
    --dino-source .cache/dinov3-repo `
    --weights .cache/dinov3-weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth `
    --out $OutDir `
    --workers 0 `
    --amp off `
    --device cuda
if ($LASTEXITCODE) { exit $LASTEXITCODE }

Write-Host "4/4 Scoring and rendering preflight check..."
& $py -m omnishield score --raw data/raw/phenikaa-mini-pilot --prepared data/prepared/phenikaa-mini-pilot --checkpoint "$OutDir/best.pt" --dino-source .cache/dinov3-repo --split val --device cuda --out "$OutDir/val_score.json"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $py -m omnishield predict --raw data/raw/phenikaa-mini-pilot --prepared data/prepared/phenikaa-mini-pilot --checkpoint "$OutDir/best.pt" --dino-source .cache/dinov3-repo --split val --device cuda --out "$OutDir/prediction"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $py -m omnishield render-replay --raw data/raw/phenikaa-mini-pilot --prepared data/prepared/phenikaa-mini-pilot --model configs/model/edge_mvp.json --split val --pred "$OutDir/prediction" --frames 4 --out "$OutDir/preflight-replay.gif"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Write-Host "Preflight complete: $OutDir"
