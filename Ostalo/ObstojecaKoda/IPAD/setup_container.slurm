#!/usr/bin/env bash
srun \
	--container-image "zironycho/pytorch:190-cu111-py37" \
	--container-save ~/Containers/IPAD.sqfs \
	--container-mount-home \
	--mem 16G \
	"conda env create -f /root/IPAD/environment.yml"
