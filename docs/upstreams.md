# Git upstreams

`anpr-imx95` is one git repository. It does not submodule the BSP. Remotes are local `git` config. To publish the same history to a second host, add another push URL:

```bash
git remote add origin git@example.com:anpr-imx95.git
git remote set-url --add --push origin git@example.com:anpr-imx95.git
git remote set-url --add --push origin git@mirror.example.com:anpr-imx95.git
```

Nothing in this repo records those URLs.

## What lives where

| Tree | Role |
| --- | --- |
| `anpr-imx95` | Application, sample config, deploy templates, host export preflight, tests, agent files |
| `imx-docker` | Docker wrapper that builds the BSP. Machine `imx95-15x15-lpddr4x-frdm`, distro `fsl-imx-wayland`, image `imx-image-full` |
| `yocto-builds` | `repo` checkout created by `imx-docker`. Do not copy it into this repo |

NXP git projects fetched by `repo` into `yocto-builds/sources`:

- `github.com/nxp-imx/imx-manifest` selects the BSP
- `github.com/nxp-imx/neutron` is the NPU userspace packaged as `neutron` 3.1.1
- `github.com/nxp-imx/tflite-neutron-delegate`, branch `lf-6.18.37_2.1.0`, builds `tensorflow-lite-neutron-delegate` 2.19.0

The eIQ Toolkit installer supplies host-side `neutron-converter`. It is not a git remote and it is not on the board rootfs. The board package `neutron` ships `/usr/lib/libNeutronDriver.so`.

Weights (`*.pt`, `*.tflite`) and calibration frames stay untracked. A Yocto layer inside this repo waits until the `/opt/anpr` install is proven.
