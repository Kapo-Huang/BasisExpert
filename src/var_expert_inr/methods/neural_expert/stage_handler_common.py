from __future__ import annotations

from collections import defaultdict

import torch.optim.lr_scheduler as lr_scheduler


class BaseTrainingStageHandler:
    loss_cls = None

    def __init__(self, stages_list, model, cfg):
        if self.loss_cls is None:
            raise RuntimeError("BaseTrainingStageHandler requires a concrete loss_cls")
        self.cfg = cfg
        self.n_samples = cfg["TRAINING"]["n_samples"]
        self.stages_list = list(stages_list)
        self.current_stage_idx = 0
        self.model = model
        self.current_stage = self.stages_list[self.current_stage_idx]

        if isinstance(cfg["TRAINING"]["lr"], dict):
            self.lr_dict = cfg["TRAINING"]["lr"]
        elif isinstance(cfg["TRAINING"]["lr"], float):
            self.lr_dict = {
                "all": cfg["TRAINING"]["lr"],
                "experts": cfg["TRAINING"]["lr"],
                "manager": cfg["TRAINING"]["lr"],
                "experts_encoder": cfg["TRAINING"]["lr"],
                "full_experts": cfg["TRAINING"]["lr"],
            }
        else:
            raise ValueError("TRAINING.lr must be either float or dict")

        if self.current_stage["params"] == "intermitent":
            self.stages_list = self.generate_intermitent_stages(self.stages_list)
            self.current_stage = self.stages_list[self.current_stage_idx]

        self.default_criterion_type = cfg["LOSS"]["loss_type"]
        self.default_criterion = self.loss_cls(
            cfg=cfg["LOSS"],
            model_name=self.cfg["MODEL"]["model_name"],
            model=self.model,
            n_experts=cfg["MODEL"]["n_experts"],
        )
        self.get_criterion()

        if "moe" in cfg["MODEL"]["model_name"]:
            self.param_dict = {
                "experts": [model.decoder.parameters],
                "manager": [model.manager_net.parameters],
            }
        else:
            self.param_dict = {"experts": [model.decoder.parameters]}

        if (
            cfg["MODEL"].get("manager_conditioning") in {"CNN", "FCN", "expert_weights"}
            and "moe" in cfg["MODEL"]["model_name"]
            and hasattr(model.manager_conditioner, "cond_encoding")
        ):
            self.param_dict["manager"].append(model.manager_conditioner.cond_encoding.parameters)

        if "learned" in cfg["MODEL"]["decoder_input_encoding"]:
            self.param_dict["experts_encoder"] = [model.decoder_input_encoding_module.parameters]
            self.param_dict["full_experts"] = [model.decoder.parameters, model.decoder_input_encoding_module.parameters]
        else:
            self.param_dict["experts_encoder"] = []
            self.param_dict["full_experts"] = []

        if (
            "moe" in cfg["MODEL"]["model_name"]
            and "learned" in cfg["MODEL"].get("manager_input_encoding", "")
            and hasattr(model, "manager_input_encoding_module")
        ):
            self.param_dict["manager"].append(model.manager_input_encoding_module.parameters)

        all_params = []
        for key, value in self.param_dict.items():
            if "encoder" not in key and key != "experts":
                all_params.extend([[item] for item in value])
        self.param_dict["all"] = all_params

    def generate_intermitent_stages(self, stages_list):
        new_stage_list = []
        stages_list = list(stages_list)
        stages_list.pop(0)
        n_substages = len(stages_list)
        step_size = 0.0
        for idx in range(n_substages):
            step_size += stages_list[idx]["end_iteration_frac"]
        n_stages = int(1.0 / step_size)
        end_iteration_frac = 0.0
        for _ in range(n_stages + 1):
            for stage in stages_list:
                end_iteration_frac += stage["end_iteration_frac"]
                new_stage_list.append(
                    {
                        "end_iteration_frac": end_iteration_frac,
                        "params": stage["params"],
                        "loss_type": stage["loss_type"],
                    }
                )
        return new_stage_list

    def get_criterion(self):
        self.cfg["LOSS"]["loss_type"] = self.current_stage["loss_type"]
        if "loss_type" in self.current_stage:
            self.criterion = self.loss_cls(
                cfg=self.cfg["LOSS"],
                model_name=self.cfg["MODEL"]["model_name"],
                model=self.model,
                n_experts=self.cfg["MODEL"]["n_experts"],
            )
        else:
            self.cfg["LOSS"]["loss_type"] = self.default_criterion_type
            self.criterion = self.default_criterion
        print(f"Currently optimizing for {self.cfg['LOSS']['loss_type']}")

    def get_trainable_params(self):
        out_list = []
        lr = self.lr_dict[self.current_stage["params"]]
        for params in self.param_dict[self.current_stage["params"]]:
            if isinstance(params, list):
                out_list.append({"params": params[0](), "lr": lr})
            else:
                out_list.append({"params": params(), "lr": lr})
        return out_list

    def get_end_iteration(self):
        return self.current_stage["end_iteration_frac"] * self.n_samples

    def get_frozen_params_dict(self):
        out_list = []
        current_params = self.param_dict[self.current_stage["params"]]
        for params in self.param_dict["all"]:
            if isinstance(current_params[0], list):
                if params not in current_params:
                    out_list.append(params[0])
            else:
                if params[0] not in current_params:
                    out_list.append(params[0])
        return out_list

    def move_to_the_next_training_stage(self, optimizer, scheduler):
        self.current_stage_idx += 1
        if self.current_stage_idx >= len(self.stages_list):
            raise RuntimeError("No more training stages")
        self.current_stage = self.stages_list[self.current_stage_idx]
        current_lr = optimizer.param_groups[0]["lr"]

        optimizer.param_groups = []
        optimizer.state = defaultdict(dict)
        for param_dict in self.get_trainable_params():
            params = list(param_dict["params"])
            optimizer.add_param_group({"params": params, "lr": current_lr})
            for param in params:
                param.requires_grad = True
        self.freeze_params()
        self.get_criterion()

    def freeze_params(self):
        for param_gen in self.get_frozen_params_dict():
            for param in param_gen():
                param.requires_grad = False

    def get_scheduler(self, optimizer):
        if self.cfg["TRAINING"]["lr_scheduler"] == "ExponentialLR":
            return lr_scheduler.ExponentialLR(optimizer, gamma=self.cfg["TRAINING"]["lr_gamma"])
        return lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
