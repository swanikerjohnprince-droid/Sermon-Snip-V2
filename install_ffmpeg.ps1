# Installs FFmpeg for Sermon-Snip: a private copy in .\local_tools\ffmpeg (no admin rights, no PATH changes needed).
# Falls back to winget if the download fails. Safe to run again.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'            # Invoke-WebRequest is painfully slow with the progress bar on
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $root 'local_tools\ffmpeg'
$url  = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'

function Test-Ffmpeg($exe) {
    if (-not (Test-Path $exe)) { return $false }
    try {
        $filters = & $exe -hide_banner -filters 2>&1 | Out-String
        $null = & $exe -hide_banner -version 2>&1
        return ($LASTEXITCODE -eq 0) -and ($filters -match '\bass\b')       # 'ass' = libass, needed to burn captions
    } catch { return $false }
}

# 1. already working?
$existing = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($existing -and (Test-Ffmpeg $existing.Source)) { Write-Host "FFmpeg is already installed and has caption support: $($existing.Source)" -ForegroundColor Green; exit 0 }
if (Test-Ffmpeg (Join-Path $dest 'bin\ffmpeg.exe')) { Write-Host "FFmpeg is already installed for Sermon-Snip: $dest" -ForegroundColor Green; exit 0 }

# 2. private download
try {
    $tmp = Join-Path $env:TEMP ('ffmpeg_' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    $zip = Join-Path $tmp 'ffmpeg.zip'
    Write-Host 'Downloading FFmpeg (about 200 MB). This takes a minute or two...'
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Write-Host 'Unpacking...'
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $found = Get-ChildItem -Path $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    if (-not $found) { throw 'ffmpeg.exe was not found inside the download.' }
    New-Item -ItemType Directory -Path (Join-Path $dest 'bin') -Force | Out-Null
    Get-ChildItem -Path $found.DirectoryName -Filter *.exe | Copy-Item -Destination (Join-Path $dest 'bin') -Force
    Remove-Item -Recurse -Force $tmp
    if (Test-Ffmpeg (Join-Path $dest 'bin\ffmpeg.exe')) {
        Write-Host "FFmpeg installed: $dest\bin" -ForegroundColor Green
        Write-Host 'Sermon-Snip will find it automatically. Nothing else to do.'
        exit 0
    }
    throw 'The downloaded FFmpeg did not pass the self-check.'
} catch {
    Write-Host "Download route failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# 3. winget fallback
if (Get-Command winget -ErrorAction SilentlyContinue) {
    Write-Host 'Trying winget (Gyan.FFmpeg)...'
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'Installed with winget. Sermon-Snip looks in winget'"'"'s folder too, so just start it again.' -ForegroundColor Green
        exit 0
    }
}
Write-Host 'Could not install FFmpeg automatically. Download the "full" build from https://www.gyan.dev/ffmpeg/builds/ and unzip it into the "local_tools\ffmpeg" folder so that local_tools\ffmpeg\bin\ffmpeg.exe exists.' -ForegroundColor Red
exit 1
