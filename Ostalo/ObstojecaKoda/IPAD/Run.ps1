# Arguments
Param(
	$Gpu = 0,
	$Norm = 2 - $Gpu,
	$Expname
)

if ($Expname) {
	$Expname += "_"
}



# Powershell/Windows setup
Set-Location $PSScriptRoot
conda activate ritnet
Start-Process -WindowStyle Hidden powershell 'conda activate ritnet; python -m visdom.server'

if (Get-Process OneDrive -ErrorAction SilentlyContinue) {
	$OneDrive = $true
	Stop-Process -Name OneDrive
}
if (Get-Process RealTimeSync -ErrorAction SilentlyContinue) {
	$RTS = $true
	Stop-Process -Name RealTimeSync
}



# Default values
$Model = "densenet"
$Datasets = "mobius_sip", "sbvpi", "smd", "sld"
$PruningBS = 2
$PrunedBS = 2
$Omegas = 0, 0.1, 0.5, 0.9, 1
$Percents = "", 50, 25

# Overwrite above values from Cfg.ps1 if it exists
if (Test-Path Cfg.ps1) {
	. .\Cfg.ps1
}

# If percent-specific batch sizes weren't defined, use existing values
if (! $PrunedBS50) {
	$PrunedBS50 = $PrunedBS
}
if (! $PrunedBS25) {
	$PrunedBS25 = $PrunedBS50
}

# If Extras wasn't defined, use default values, which differ per model
if (!(Test-Path Variable:Extras)) {
	if ($Model -eq "densenet") {
		$Extras = "", "--prune conv", "--channelsUseWeightsOnly"
	} else {
		[array]$Extras = ""
	}
}



# Run experiments
foreach ($Dataset in $Datasets) {
	foreach ($Omega in $Omegas) {
		foreach ($Extra in $Extras) {
			$Exp = $Expname
			if ($Norm -ne 2) {
				$Exp += "l${Norm}_"
			}
			switch ($Extra) {
				"--prune conv" {
					$Exp += "3x3_only_"
				}
				"--channelsUseWeightsOnly" {
					$Exp += "1x1_weights_only_"
				}
			}

			$PruningExp = "matej_${Dataset}_${Model}/${Exp}pruning_omega${Omega}"
			if (Test-Path "logs\$PruningExp") {
				Write-Output "$PruningExp already exists, skipping"
			} else {
				python train_with_pruning_combined.py `
					--dataset "eyes_$Dataset" `
					--model "$Model" `
					--bs $PruningBS `
					--workers 0 `
					--norm $Norm `
					--expname "$PruningExp" `
					--resume "logs/matej_${Dataset}_${Model}/original/models/${Model}_best.pkl" `
					--pruningGpus $Gpu `
					$Extra
			}
			if (! $?) {
				# Exit if there was an error
				$Exit = $LastExitCode
				Remove-Item "logs\$PruningExp" -Recurse -Confirm
				exit $Exit
			}

			foreach ($Percent in $Percents) {
				if ($Percent) {
					$PercentExp = "_pruned$Percent"
					$PercentModel = $Percent
				} else {
					$PercentExp = ""
					$PercentModel = "final"
				}

				$FinalExp = "matej_${Dataset}_${Model}/${Exp}final_omega${Omega}${PercentExp}"
				if (Test-Path "logs\$FinalExp") {
					Write-Output "$FinalExp already exists, skipping"
					continue
				}
				python train_pruned_model.py `
					--dataset "eyes_$Dataset" `
					--model "$Model" `
					--bs $(Get-Variable "PrunedBS${Percent}" -ValueOnly) `
					--workers 0 `
					--expname "$FinalExp" `
					--pruningGpus $Gpu `
					--resume "logs/${PruningExp}/models/${Model}_${PercentModel}.pt"
				if (! $?) {
					# Exit if there was an error
					$Exit = $LastExitCode
					Remove-Item "logs\$FinalExp" -Recurse -Confirm
					exit $Exit
				}
			}
		}
	}
}



# Restore OneDrive and RTS if they were closed by this script
if ($OneDrive) {
	Start-Process "${Env:LOCALAPPDATA}\Microsoft\OneDrive\OneDrive.exe" -ArgumentList "/background"
}
if ($RTS) {
	Start-Process "${Env:USERPROFILE}\Documents\File Sync\EyeZ Main.ffs_real"
}