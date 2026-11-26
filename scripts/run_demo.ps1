param(
    [switch]$WithAI,
    [string]$Checkpoint = "weights/mvp_v5_best.pt",
    [string]$Split = "all",
    [int]$Port = 8765,
    [double]$Threshold = 0.50
)

$py = "C:\Users\Hoang\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $py)) {
    throw "Python 3.11 CUDA runtime not found at $py"
}

$demoArgs = @(
    "-m", "omnishield", "demo",
    "--raw", "data/raw/phenikaa-mini-mvp",
    "--prepared", "data/prepared/phenikaa-mini-mvp",
    "--split", $Split,
    "--device", "cuda",
    "--port", $Port,
    "--threshold", $Threshold,
    "--open"
)

if ($WithAI) {
    $demoArgs += @("--checkpoint", $Checkpoint, "--dino-source", ".cache/dinov3-repo")
    if (Test-Path "runs/calib_lidar_profile.json") {
        $demoArgs += @("--calibrated-rig", "runs/calib_lidar_profile.json")
    }
} else {
    $demoArgs += @("--no-ai", "--model", "configs/model/edge_mvp.json")
    if (Test-Path "runs/calib_camera_only_profile.json") {
        $demoArgs += @("--calibrated-rig", "runs/calib_camera_only_profile.json")
    }
}

& $py @demoArgs
