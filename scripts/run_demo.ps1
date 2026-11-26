param(
    [string]$Checkpoint = "weights/mvp_v5_best.pt",
    [string]$Split = "all",
    [int]$Port = 8765,
    [double]$Threshold = 0.50
)

$py = "C:\Users\Hoang\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $py)) {
    throw "Python 3.11 CUDA runtime not found at $py"
}

& $py -m omnishield demo `
    --raw data/raw/phenikaa-mini-mvp `
    --prepared data/prepared/phenikaa-mini-mvp `
    --checkpoint $Checkpoint `
    --dino-source .cache/dinov3-repo `
    --split $Split `
    --device cuda `
    --port $Port `
    --threshold $Threshold `
    --open
