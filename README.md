# hidden-currents

## What is it?

A PINN that reads daily sea surface temperature (SST) maps and works out the current that deforms them. It never sees a velocity. It learns a stream function ψ (so the field is divergence free), a diffusivity κ and a small source term, under the constraint

    ∂T/∂t + u·∇T = κ ΔT + S        u = -∂ψ/∂y,  v = ∂ψ/∂x

The inferred currents are checked against HF radars, which actually measure velocity. A Horn–Schunck optical flow is the dumb baseline.

Two ERDDAP servers, no account needed:

- SST: NASA JPL MUR 0.01°, daily, via NOAA CoastWatch, `jplMURSST41`. Pulled at stride 5, so 0.05°, about the radar grid
- HF radar: EUSKOOS totals (SE Bay of Biscay, 29 × 33 cells), hourly since December 2018, via EMODnet Physics, `EUHFR_NRTcurrent_HFR-EUSKOOS-Total`

The SST box is the radar box plus 0.4°, read from the radar metadata, so pointing `--radar-id` at another HF radar network moves everything with it. Default window is October 2020.

Ifremer's own ERDDAP used to carry both (ODYSSEA SST and a mirror of the radar). In November 2026 it holds 55 datasets, mostly Argo, and the radar mirror lives under a hashed ID that changes on every re-harvest. Hence the two other servers.

A synthetic twin (`synth.py`) runs everything offline and checks that the PINN recovers a known field.

`web/` is a static viewer: SST, PINN arrows against radar arrows, metrics, training curve.

![viewer](web/screenshot.png)

On the synthetic twin (4000 steps, CPU): correlation 0.84 on both u and v, angle error 18°, RMSE 0.049 m/s for a mean speed of 0.09 m/s. Optical flow gets 0.40 / 49° / 0.099 on the same points.

## How to run it

    pip install -r requirements.txt

Synthetic twin, end to end:

    python synth.py
    python pinn.py --data data/synth.npz --out runs/synth --steps 4000
    python evaluate.py --run runs/synth --data data/synth.npz
    python -m http.server -d web 8000

Real data. Check what the servers have first, then pull:

    python fetch.py --info
    python fetch.py --start 2020-10-01 --end 2020-10-21
    python pinn.py --data data/sst_2020-10-01_2020-10-21.nc --out runs/oct2020 --steps 12000
    python evaluate.py --run runs/oct2020 --data data/sst_2020-10-01_2020-10-21.nc --radar data/radar_2020-10-01_2020-10-21.nc

Then http://localhost:8000.

On CPU, 4000 steps with batch 2048 take ~7 min. On a GPU it is a minute. Use `--batch 4096` as soon as memory allows.

Radar days with fewer than 12 valid hours are dropped (a short mean is mostly tide), and so are cells with GDOP above 2, the usual EU HFR cut.

Tests, including the real-data path on files shaped like the two servers' output:

    pytest -q

## TODO

- κ comes out at ~22 m²/s for a true 50 in the twin: the source term S eats part of the diffusion. Regularize S harder, or freeze it for the first half of training
- MUR is a blended analysis: on cloudy days it is interpolation, not observation. Weight the data loss by MUR's `analysis_error`
- use the radar QC flags on top of GDOP
- in summer S absorbs too much (heat flux). Constrain it with a climatology or make it spatially uniform
- L-BFGS at the end of training
- try chlorophyll as the tracer (a bloom is a reaction term, the real test)
- the 3-week window is a guess, sweep it
- GeoJSON export of the currents for QGIS
