param (
    [Parameter(Mandatory=$true, HelpMessage="Team ID (e.g., team0042)")][string]$TeamId,
    [Parameter(Mandatory=$true, HelpMessage="Version number (e.g., 1)")][string]$Version
)

$ImageName = "${TeamId}:v${Version}"
$TarName = "${TeamId}_v${Version}.tar"
$ArchiveName = "${TeamId}_v${Version}.tar.gz"

Write-Host "Building Docker image: $ImageName for linux/amd64..." -ForegroundColor Cyan
docker build --platform linux/amd64 -t $ImageName .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed."
    exit $LASTEXITCODE
}

Write-Host "Saving image to $TarName..." -ForegroundColor Cyan
docker save -o $TarName $ImageName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker save failed."
    exit $LASTEXITCODE
}

Write-Host "Compressing to $ArchiveName using Python gzip..." -ForegroundColor Cyan
python -c "import gzip, shutil; f_in = open('$TarName', 'rb'); f_out = gzip.open('$ArchiveName', 'wb'); shutil.copyfileobj(f_in, f_out); f_in.close(); f_out.close()"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compression failed."
    exit $LASTEXITCODE
}

Write-Host "Cleaning up temporary tar file..." -ForegroundColor Cyan
Remove-Item $TarName

Write-Host "Done! The submission archive is ready: $ArchiveName" -ForegroundColor Green
Get-Item $ArchiveName | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length / 1MB, 2)}}
