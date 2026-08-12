# Build the Windows desktop package with PyInstaller.
# Run from repo root on Windows:
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Installing package and build dependencies"
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
python -m pip install -e ".[build]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

$Dist = Join-Path $Root "dist"
$Build = Join-Path $Root "build"
if (Test-Path $Dist) { Remove-Item -Recurse -Force $Dist }
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }

Write-Host "==> Running PyInstaller"
python -m PyInstaller `
  --noconfirm `
  --clean `
  --workpath $Build `
  --distpath $Dist `
  (Join-Path $Root "desktop\wb-analyzer.spec")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$AppDir = Join-Path $Dist "WB-Analyzer"
$ExePath = Join-Path $AppDir "WB-Analyzer.exe"
if (-not (Test-Path $ExePath)) {
  throw "Build failed: $ExePath not found"
}

# desktop/wb-analyzer.spec prunes Qt DLLs and plugins by name (_DROP_KEYS) to keep
# the bundle small. That filter is pattern-based, so an over-broad pattern can
# silently strip something the app needs. Assert the essentials survived before
# we ship a zip; qwindows.dll in particular is the Qt platform plugin, without
# which the exe dies with "could not find or load the Qt platform plugin windows".
Write-Host "==> Verifying bundle integrity"
$Required = @("python3*.dll", "Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "qwindows.dll")
$Missing = @()
foreach ($Pattern in $Required) {
  $Found = Get-ChildItem -Path $AppDir -Recurse -File -Filter $Pattern -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if ($Found) {
    Write-Host "  ok      $Pattern -> $($Found.FullName.Substring($AppDir.Length + 1))"
  } else {
    Write-Host "  MISSING $Pattern"
    $Missing += $Pattern
  }
}
if ($Missing.Count -gt 0) {
  throw "Bundle is missing required runtime files: $($Missing -join ', ')"
}

$SizeMb = [math]::Round(
  ((Get-ChildItem -Path $AppDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 1)
Write-Host "  Bundle size: $SizeMb MB"

$ZipPath = Join-Path $Dist "WB-Analyzer-Windows.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path $AppDir -DestinationPath $ZipPath

Write-Host ""
Write-Host "Build complete."
Write-Host "  App folder: $AppDir"
Write-Host "  Launcher:   $ExePath"
Write-Host "  Zip:        $ZipPath"
Write-Host ""
Write-Host "Distribute the zip, or run WB-Analyzer.exe inside the folder."
