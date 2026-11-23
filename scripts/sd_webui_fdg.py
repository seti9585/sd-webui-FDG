import math
import os

import gradio as gr
import torch
from modules import scripts, shared


def fdg_prediction(c, u, high, low, dtype):
    from kornia.geometry.transform import pyrdown, pyrup
    from torch.nn.functional import pad

    if c.shape != u.shape or c.ndim != 4 or c.device != u.device or c.dtype != u.dtype:
        raise ValueError("FDG requires matching BCHW predictions")
    if not c.is_floating_point():
        raise ValueError("FDG requires floating-point predictions")
    h, w = c.shape[-2:]
    if min(h, w) < 3:
        raise ValueError("FDG spatial dimensions must be at least 3")
    ph, pw = (1 << (h - 1).bit_length()) - h, (1 << (w - 1).bit_length()) - w
    with torch.autocast(device_type=c.device.type, enabled=False):
        pair = torch.cat((c, u)).float()
        if (ph and pw) or h % 2 or w % 2:
            pair = pad(pair, (0, pw, 0, ph), mode="reflect")
        coarse = pyrdown(pair)
        detail = pair - pyrup(coarse)
        ch, uh = detail.chunk(2)
        cl, ul = coarse.chunk(2)
        result = (uh + high * (ch - uh) + pyrup(ul + low * (cl - ul)))
        result = result[..., :h, :w].to(dtype)
        if not torch.isfinite(result).all():
            raise ValueError("FDG produced a non-finite prediction")
        return result



def fdg_delta_prediction(c, u, backend_cfg, high, low):
    """Add high-band guidance to the backend's plain CFG result."""
    if c.shape != u.shape or c.ndim != 4 or c.device != u.device or c.dtype != u.dtype:
        raise ValueError("FDG requires matching BCHW predictions")
    if (backend_cfg.shape != c.shape or backend_cfg.device != c.device
            or not c.is_floating_point() or not backend_cfg.is_floating_point()):
        raise ValueError("FDG requires matching floating-point backend output")
    if not all(math.isfinite(v) and 0 <= v <= 40 for v in (low, high)):
        raise ValueError("Invalid FDG scale")
    h, w = c.shape[-2:]
    if min(h, w) < 3:
        raise ValueError("FDG spatial dimensions must be at least 3")
    # Preserve the exact backend tensor when no extra guidance is requested.
    if high == low:
        if not torch.isfinite(backend_cfg).all():
            raise ValueError("FDG received a non-finite backend output")
        return backend_cfg
    from kornia.geometry.transform import pyrdown, pyrup
    from torch.nn.functional import pad

    ph = (1 << (h - 1).bit_length()) - h
    pw = (1 << (w - 1).bit_length()) - w
    with torch.autocast(device_type=c.device.type, enabled=False):
        delta = c.float() - u.float()
        if (ph and pw) or h % 2 or w % 2:
            delta = pad(delta, (0, pw, 0, ph), mode="reflect")
        delta_high = delta - pyrup(pyrdown(delta))
        result = (backend_cfg.float() + (high - low) * delta_high[..., :h, :w])
        result = result.to(backend_cfg.dtype)
        if not torch.isfinite(result).all():
            raise ValueError("FDG produced a non-finite prediction")
        return result



def fdg_projected_prediction(c, u, high, low, dtype,
                             eta_low=1.0, eta_high=0.5, backend_cfg=None):
    """Apply Appendix D's parallel projection weights separately per band."""
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (eta_low, eta_high)):
        raise ValueError("FDG projection weights must be between 0 and 1")
    if not all(math.isfinite(v) and 0 <= v <= 40 for v in (low, high)):
        raise ValueError("Invalid FDG scale")
    # Neutral weights must preserve the selected existing mode bit for bit.
    if eta_low == 1 and eta_high == 1:
        if backend_cfg is not None:
            return fdg_delta_prediction(c, u, backend_cfg, high, low)
        return fdg_prediction(c, u, high, low, dtype)
    if (c.shape != u.shape or c.ndim != 4 or c.device != u.device
            or c.dtype != u.dtype or not c.is_floating_point()):
        raise ValueError("FDG requires matching floating-point BCHW predictions")
    h, w = c.shape[-2:]
    if min(h, w) < 3:
        raise ValueError("FDG spatial dimensions must be at least 3")
    from kornia.geometry.transform import pyrdown, pyrup
    from torch.nn.functional import pad, normalize

    ph = (1 << (h - 1).bit_length()) - h
    pw = (1 << (w - 1).bit_length()) - w
    with torch.autocast(device_type=c.device.type, enabled=False):
        pair = torch.cat((c, u)).float()
        if (ph and pw) or h % 2 or w % 2:
            pair = pad(pair, (0, pw, 0, ph), mode="reflect")
        coarse = pyrdown(pair)
        detail = pair - pyrup(coarse)
        ch, uh = detail.chunk(2)
        cl, ul = coarse.chunk(2)
        guided = []
        for pc, pu, scale, eta in ((ch, uh, high, eta_high),
                                    (cl, ul, low, eta_low)):
            diff = pc - pu
            if eta != 1:
                # Match the paper: FP64 projection, per-sample C/H/W norm,
                # and separate casts for parallel and orthogonal components.
                direction = normalize(pc.double(), dim=(-1, -2, -3))
                diff64 = diff.double()
                parallel = (diff64 * direction).sum(
                    dim=(-1, -2, -3), keepdim=True) * direction
                orthogonal = diff64 - parallel
                diff = eta * parallel.to(diff.dtype) + orthogonal.to(diff.dtype)
            guided.append(pc + (scale - 1) * diff)
        result = (guided[0] + pyrup(guided[1]))[..., :h, :w].to(dtype)
        if not torch.isfinite(result).all():
            raise ValueError("FDG projection produced a non-finite prediction")
        return result

class FDGScript(scripts.Script):
    def title(self):
        return "FDG"

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("FDG", open=False):
            enabled = gr.Checkbox(label="Enable FDG", value=False)
            use_backend_low = gr.Checkbox(label="Use main CFG as Low CFG", value=True)
            low = gr.Slider(0, 40, value=5, step=0.5, label="Low CFG", interactive=False)
            high = gr.Slider(0, 40, value=10, step=0.5, label="High CFG")
            use_backend_low.change(
                fn=lambda use_main: gr.update(interactive=not use_main),
                inputs=[use_backend_low], outputs=[low])
            projection = gr.Checkbox(label="Enable band-wise APG projection", value=False)
            eta_low = gr.Slider(0, 1, value=1, step=0.05,
                               label="Low parallel weight (1 = unchanged, 0 = remove)",
                               interactive=False)
            eta_high = gr.Slider(0, 1, value=0.5, step=0.05,
                                label="High parallel weight (1 = unchanged, 0 = remove)",
                                interactive=False)
            projection.change(
                fn=lambda active: (gr.update(interactive=active), gr.update(interactive=active)),
                inputs=[projection], outputs=[eta_low, eta_high])
            apply_base = gr.Checkbox(label="Apply FDG to Base", value=True)
            apply_hires = gr.Checkbox(label="Apply FDG to Hires", value=False)
        # New controls are appended to preserve earlier API argument order.
        return [enabled, low, high, use_backend_low, projection, eta_low, eta_high, apply_base, apply_hires]

    def process(self, p, enabled, low, high, use_backend_low=False,
                projection=False, eta_low=1.0, eta_high=0.5, apply_base=True, apply_hires=False):
        if not enabled:
            return
        try:
            p.fdg_debug = int(os.environ.get("SD_WEBUI_SETI_DEBUG", "0")) >= 1
        except ValueError:
            p.fdg_debug = False
        p.fdg_calls = 0
        p.fdg_pass_records = []
        p.fdg_registered = False
        p.fdg_skipped = False

    def process_before_every_sampling(self, p, enabled, low, high,
                                      use_backend_low=False, projection=False,
                                      eta_low=1.0, eta_high=0.5, apply_base=True, apply_hires=False, **kwargs):
        if not enabled:
            return
        pass_name = "Hires" if getattr(p, "is_hr_pass", False) else "Base"
        patcher = p.sd_model.forge_objects.unet
        # Remove only callbacks owned by this processing object, if a host reuses it.
        callbacks = patcher.model_options.get("sampler_post_cfg_function", [])
        if any(getattr(fn, "_fdg_owner", None) is p for fn in callbacks):
            patcher = patcher.clone()
            patcher.model_options = dict(patcher.model_options)
            remaining = [fn for fn in callbacks if getattr(fn, "_fdg_owner", None) is not p]
            if remaining:
                patcher.model_options["sampler_post_cfg_function"] = remaining
            else:
                patcher.model_options.pop("sampler_post_cfg_function", None)
            p.sd_model.forge_objects.unet = patcher
        record = {"pass": pass_name, "calls": 0, "status": "Disabled"}
        p.fdg_pass_records.append(record)
        prefix = f"FDG {pass_name}"
        selected = apply_hires if pass_name == "Hires" else apply_base
        p.extra_generation_params[prefix + " requested"] = "ON" if selected else "OFF"
        if not selected:
            if p.fdg_debug:
                print(f"[FDG {pass_name}] Disabled")
            return
        pass_cfg = float(getattr(p, "hr_cfg", p.cfg_scale)) if pass_name == "Hires" else float(p.cfg_scale)
        effective_low = pass_cfg if use_backend_low else float(low)
        high = float(high)
        active_projection = bool(projection and (eta_low != 1 or eta_high != 1))
        if projection and not all(math.isfinite(v) and 0 <= v <= 1 for v in (eta_low, eta_high)):
            raise ValueError("FDG projection weights must be between 0 and 1")
        source = ("Hires CFG" if pass_name == "Hires" else "Main CFG") if use_backend_low else "Independent Low"
        mode = ("Band-wise projection reconstruction" if active_projection else
                ("Backend CFG + high-frequency delta" if use_backend_low else "Independent band reconstruction"))
        p.extra_generation_params[prefix + " requested"] = f"Low {effective_low} / High {high}"
        p.extra_generation_params[prefix + " Low source"] = source
        p.extra_generation_params[prefix + " calculation"] = mode
        if projection:
            p.extra_generation_params[prefix + " projection requested"] = f"Low parallel {eta_low} / High parallel {eta_high}"
        opts = patcher.model_options
        conflicts = {
            "CFG <= 1": pass_cfg <= 1,
            "epsilon scaling": getattr(shared.opts, "epsilon_scaling_enabled", False),
            "Skip Early CFG": getattr(shared.opts, "skip_early_cond", 0),
            "NGMS": getattr(shared.opts, "s_min_uncond", 0),
        }
        for key in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function"):
            conflicts[key] = opts.get(key)
        reasons = [name for name, active in conflicts.items() if active]
        if reasons:
            record["status"] = "Skipped: " + ", ".join(reasons)
            p.extra_generation_params[prefix + " skipped"] = ", ".join(reasons)
            print(f"[FDG {pass_name}] " + record["status"])
            return
        if not all(math.isfinite(v) and 0 <= v <= 40 for v in (effective_low, high)):
            raise ValueError("Invalid FDG scale")
        # Resolve dependencies before sampling; the exact pass-through needs none.
        if active_projection or not (use_backend_low and high == effective_low):
            try:
                from kornia.geometry.transform import pyrdown, pyrup
            except ImportError as exc:
                raise RuntimeError("FDG requires Kornia with pyrdown and pyrup") from exc

        def apply_fdg(args):
            if p.fdg_debug and record["calls"] == 0:
                for name in ("cond_denoised", "uncond_denoised"):
                    tensor = args.get(name)
                    detail = "missing" if tensor is None else f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                    print(f"[FDG {pass_name}] {name}: {detail}")
            cond = args["cond"]
            if (not args["uncond"] or not cond or len(cond) != 1
                    or cond[0].get("strength", 1) != 1
                    or args["uncond_denoised"] is None):
                raise ValueError("FDG requires plain conditional and unconditional predictions")
            if use_backend_low and float(args["cond_scale"]) != effective_low:
                raise ValueError("FDG main CFG changed during sampling")
            if active_projection:
                result = fdg_projected_prediction(
                    args["cond_denoised"], args["uncond_denoised"],
                    high, effective_low, args["denoised"].dtype, eta_low, eta_high)
            elif use_backend_low:
                result = fdg_delta_prediction(
                    args["cond_denoised"], args["uncond_denoised"],
                    args["denoised"], high, effective_low)
            else:
                result = fdg_prediction(
                    args["cond_denoised"], args["uncond_denoised"],
                    high, effective_low, args["denoised"].dtype)
            record["calls"] += 1
            p.fdg_calls += 1
            return result

        apply_fdg._fdg_owner = p
        record["status"] = "Registered"
        clone = patcher.clone()
        clone.set_model_sampler_post_cfg_function(apply_fdg)
        p.sd_model.forge_objects.unet = clone
        if p.fdg_debug:
            print(f"[FDG {pass_name}] Low source={source}; Calculation={mode}; Low={effective_low} High={high}")
            print(f"[FDG {pass_name}] Post-CFG hook registered")
            if projection:
                print(f"[FDG {pass_name}] Band projection: Low parallel={eta_low} High parallel={eta_high}")
            if use_backend_low and high == effective_low and not active_projection:
                print(f"[FDG {pass_name}] Equal scales: exact backend pass-through")

    def postprocess(self, p, processed, enabled, low, high, use_backend_low=False,
                    projection=False, eta_low=1.0, eta_high=0.5, apply_base=True, apply_hires=False):
        if enabled and getattr(p, "fdg_debug", False):
            for record in getattr(p, "fdg_pass_records", []):
                print(f"[FDG {record['pass']}] Predictions returned: {record['calls']} ({record['status']})")
            print(f"[FDG] Predictions returned: {getattr(p, 'fdg_calls', 0)}")
