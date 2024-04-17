Param(
	$Datasets = "D:\EyeZ\Segmentation\Sclera\Datasets\Matic RITnet",
	$Results = $Datasets.Replace('Datasets', 'Results')
)

# This script needs to be run as administrator to be able to make symlinks
$MyWindowsID = [System.Security.Principal.WindowsIdentity]::GetCurrent();
$MyWindowsPrincipal = New-Object System.Security.Principal.WindowsPrincipal($MyWindowsID);
$AdminRole = [System.Security.Principal.WindowsBuiltInRole]::Administrator;
if ($MyWindowsPrincipal.IsInRole($AdminRole)) {
    $Host.UI.RawUI.WindowTitle = $MyInvocation.MyCommand.Definition + "(Elevated)";
    $Host.UI.RawUI.BackgroundColor = "DarkBlue";
    Clear-Host;
}
else {
    $NewProcess = New-Object System.Diagnostics.ProcessStartInfo "PowerShell";
    $NewProcess.Arguments = "& '" + $Script:MyInvocation.MyCommand.Path + "' '$Datasets' '$Results'"
    $NewProcess.Verb = "RunAs";
    [System.Diagnostics.Process]::Start($NewProcess);
    Exit;
}

Set-Location $PSScriptRoot

# Datasets
foreach ($Dir in Get-ChildItem ${Datasets} -Filter "eyes*") {
	if (!(Test-Path $Dir.Name)) {
        New-Item -ItemType SymbolicLink -Path $Dir.Name -Target $Dir.FullName
    }
}

# Results
if (!(Test-Path logs)) {
    New-Item -ItemType SymbolicLink -Path logs -Target $Results
}
if (!(Test-Path test)) {
    if (!(Test-Path "$Results\test")) {
        New-Item -ItemType Directory -Path "$Results\test"
    }
    New-Item -ItemType SymbolicLink -Path test -Target "$Results\test"
}

# UNet teachers (needed for distillation and LeGR)
foreach ($Dataset in "mobius_sip", "sbvpi", "smd", "sld") {
    $Dir = "logs\matej_${Dataset}_unet"
    if (!(Test-Path "$Dir")) {
        continue
    }
    if (!(Test-Path "$Dir\teacher\models")) {
        New-Item -ItemType Directory -Path "$Dir\teacher\models"
    }
    foreach ($Model in "best", "final") {
        if (!(Test-Path "$Dir\teacher\models\teacher_${Model}.pkl")) {
            New-Item -ItemType HardLink -Path "$Dir\teacher\models\teacher_${Model}.pkl" -Target "$Dir\original\models\unet_${Model}.pkl"
        }
    }
}

# LeGR model directory (needed for loading LeGR models)
if (Test-Path "..\LeGR") {
    if (!(Test-Path "model")) {
        New-Item -ItemType SymbolicLink -Path "model" -Target "..\LeGR\model"
    }
    if (!(Test-Path "..\LeGR\ckpt")) {
        New-Item -ItemType SymbolicLink -Path "..\LeGR\ckpt" -Target "${Results}\LeGR"
    }
}
