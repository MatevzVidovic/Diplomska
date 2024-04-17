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
$LeGRBS = 2
$PrunedBS = 2
$Omegas = 0, 0.1, 0.5, 0.9, 1
$Percents = "", 50, 25

# Overwrite above values from Run.cfg if it exists
if (Test-Path "..\IPAD\Cfg.ps1") {
	. "..\IPAD\Cfg.ps1"
}
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



# Run experiments
foreach ($Dataset in $Datasets) {
	foreach ($Omega in $Omegas) {
		foreach ($Percent in $Percents) {
			$Exp = "matej_${Dataset}_${Model}"
			$Prefix = $Expname
			if ($Norm -ne 2) {
				$Prefix += "l${Norm}_"
			}
			$Suffix = "omega${Omega}"
			if ($Percent) {
				$Suffix += "_pruned${Percent}"
				$PercentVar = "PrunedBS${Percent}"
			}
			else {
				$Percent = 75
				$PercentVar = "PrunedBS"
			}
			$FullExp = "${Exp}/${Prefix}${Suffix}"

			Set-Location "..\LeGR"
			if (Test-Path "ckpt/${FullExp}_bestarch_init.pt") {
				Write-Output "$FullExp already exists, skipping"
			} else {
				if ("$Model" -eq "unet") {
					$Pruner = "FilterPrunerUNet"
				} else {
					$Pruner = "FilterPrunerRitnet"
				}
				python legr.py `
					--name "$FullExp" `
					--dataset "eyes_$Dataset" `
					--model "..\IPAD\logs\${Exp}\teacher\models\teacher_best.pkl" `
					--batch_size $LeGRBS `
					--workers 0 `
					--omega $Omega `
					--prune_away $Percent `
					--pruner $Pruner `
					--gpu $Gpu `
					--rank_type "l${Norm}_combined"
				if (! $?) {
					# Exit if there was an error
					exit $LastExitCode
				}
			}

			Set-Location "..\IPAD"
			if (Test-Path "logs\${Exp}\legr_${Prefix}${Suffix}") {
				Write-Output "${Prefix}${Suffix} already exists, skipping"
				continue
			}
			python train_pruned_model.py `
				--expname "${Exp}/legr_${Prefix}${Suffix}" `
				--dataset "eyes_$Dataset" `
				--model "$Model" `
				--resume "..\LeGR\ckpt\${FullExp}_bestarch_init.pt" `
				--bs $(Get-Variable $PercentVar -ValueOnly) `
				--workers 0 `
				--pruningGpus $Gpu
			if (! $?) {
				# Exit if there was an error
				$Exit = $LastExitCode
				Remove-Item "logs\${Exp}\legr_${Prefix}${Suffix}" -Recurse -Confirm
				exit $Exit
			}
		}
	}
}
