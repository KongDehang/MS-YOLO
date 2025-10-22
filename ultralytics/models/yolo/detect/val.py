# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (list[Any]): List for storing ground truth labels for hybrid saving.
        jdict (list[dict[str, Any]]): List for storing JSON detection results.
        stats (dict[str, list[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (list[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

        # Support for secondary classification head (nc2)
        self.names2 = getattr(model, "names2", None)
        self.nc2 = len(self.names2) if self.names2 else 0

        # If model doesn't have names2 (e.g., fused model), try to get it from data.yaml
        if self.nc2 == 0 and "nc2" in self.data and self.data["nc2"] > 0:
            self.nc2 = self.data["nc2"]
            # Create default names2 if not in data.yaml
            if "names2" in self.data:
                self.names2 = self.data["names2"]
            else:
                # Create default material names
                self.names2 = {i: str(i) for i in range(self.nc2)}
            LOGGER.info(f"Restored names2 from data.yaml: nc2={self.nc2}")

        self.confusion_matrix2 = None
        self.metrics2 = None  # Secondary classification metrics
        self.stats2 = None  # Secondary classification stats
        if self.nc2 > 0:
            self.confusion_matrix2 = ConfusionMatrix(
                names=self.names2, save_matches=self.args.plots and self.args.visualize
            )
            self.metrics2 = DetMetrics(names=self.names2)  # Independent metrics for cls2

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'extra' tensors.
        """
        # Pass correct nc to NMS so it can preserve extra channels (nc2)
        # nc should be the number of primary classes, not 0
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,  # Use self.nc instead of 0 to preserve extra channels
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        # Extract cls2 from extra channels if nc2 is present
        results = []
        for x in outputs:
            result = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
            # If nc2 exists, extract cls2 predictions (argmax of nc2 classes in extra)
            if self.nc2 > 0 and x.shape[1] > 6:
                cls2_scores = x[:, 6 : 6 + self.nc2]  # Get nc2 class scores
                result["cls2"] = cls2_scores.argmax(dim=1)  # Get predicted class index
                result["cls2_conf"] = cls2_scores.max(dim=1).values  # Get confidence for cls2
            results.append(result)  # Add result to list
        return results

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes

        result = {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

        # Add cls2 if present in batch
        if "cls2" in batch and self.nc2 > 0:
            cls2 = batch["cls2"][idx].squeeze(-1)
            result["cls2"] = cls2

        return result

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            batch (dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            # # Debug: Log cls2 presence (only first batch)
            # if si == 0 and self.seen == 1 and self.nc2 > 0:
            #     LOGGER.info(f"DEBUG: cls2 in batch keys: {'cls2' in batch}")
            #     LOGGER.info(f"DEBUG: cls2 in pbatch keys: {'cls2' in pbatch}")
            #     LOGGER.info(f"DEBUG: cls2 in predn keys: {'cls2' in predn}")
            #     if 'cls2' in batch:
            #         LOGGER.info(f"DEBUG: batch cls2 shape: {batch['cls2'].shape}")
            #         LOGGER.info(f"DEBUG: batch cls2 values: {batch['cls2'][:10]}")  # First 10
            #     if 'cls2' in predn:
            #         LOGGER.info(f"DEBUG: predn cls2 shape: {predn['cls2'].shape}")
            #         LOGGER.info(f"DEBUG: predn cls2 unique: {predn['cls2'].unique()}")
            #     if 'cls2' in pbatch:
            #         LOGGER.info(f"DEBUG: pbatch cls2 shape: {pbatch['cls2'].shape}")
            #         LOGGER.info(f"DEBUG: pbatch cls2 values: {pbatch['cls2']}")
            #         LOGGER.info(f"DEBUG: pbatch cls shape: {pbatch['cls'].shape}")
            #         LOGGER.info(f"DEBUG: batch_idx filter sum: {(batch['batch_idx'] == si).sum()}")

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            # Update metrics for secondary classification (cls2) if present
            if self.metrics2 is not None and "cls2" in predn and "cls2" in pbatch:
                cls2 = pbatch["cls2"].cpu().numpy()
                predn2 = {k: v for k, v in predn.items()}
                predn2["cls"] = predn["cls2"]  # Use cls2 as the main cls for metrics calculation
                predn2["conf"] = predn.get("cls2_conf", predn["conf"])  # Use cls2 confidence
                pbatch2 = {k: v for k, v in pbatch.items()}
                pbatch2["cls"] = pbatch["cls2"]  # Use cls2 as ground truth

                # # Debug: Log cls2 update_stats (only first batch)
                # if si == 0 and self.seen == 1:
                #     LOGGER.info(f"DEBUG: Calling metrics2.update_stats")
                #     LOGGER.info(f"DEBUG: predn2 cls shape: {predn2['cls'].shape}, unique: {predn2['cls'].unique()}")
                #     LOGGER.info(f"DEBUG: pbatch2 cls shape: {pbatch2['cls'].shape}, values: {pbatch2['cls']}")
                #     LOGGER.info(f"DEBUG: predn2 bboxes shape: {predn2['bboxes'].shape}")
                #     LOGGER.info(f"DEBUG: pbatch2 bboxes shape: {pbatch2['bboxes'].shape}")

                self.metrics2.update_stats(
                    {
                        **self._process_batch(predn2, pbatch2),
                        "target_cls": cls2,
                        "target_img": np.unique(cls2),
                        "conf": np.zeros(0) if no_pred else predn2["conf"].cpu().numpy(),
                        "pred_cls": np.zeros(0) if no_pred else predn2["cls"].cpu().numpy(),
                    }
                )

            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                # Process second classification head if exists
                if self.confusion_matrix2 is not None and "cls2" in predn and "cls2" in pbatch:
                    # Create separate batches for cls2
                    predn2 = {k: v for k, v in predn.items()}
                    predn2["cls"] = predn.get("cls2", predn["cls"])
                    pbatch2 = {k: v for k, v in pbatch.items()}
                    pbatch2["cls"] = pbatch.get("cls2", pbatch["cls"])
                    self.confusion_matrix2.process_batch(predn2, pbatch2, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
                # Plot second confusion matrix if exists
                if self.confusion_matrix2 is not None:
                    self.confusion_matrix2.plot(
                        save_dir=self.save_dir,
                        normalize=normalize,
                        on_plot=self.on_plot,
                        names_suffix="_cls2",  # Add suffix to distinguish from first matrix
                    )
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

        # Process metrics2 if present
        if self.metrics2 is not None:
            self.metrics2.speed = self.speed
            if self.confusion_matrix2 is not None:
                self.metrics2.confusion_matrix = self.confusion_matrix2
            self.metrics2.save_dir = self.save_dir

    def get_stats(self) -> dict[str, Any]:
        """
        Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        stats = self.metrics.results_dict

        # Get cls1 fitness (original)
        fitness1 = stats.get("fitness", 0.0)

        # Process metrics2 if present and has data
        if self.metrics2 is not None and len(self.metrics2.stats.get("tp", [])) > 0:
            try:
                # LOGGER.info(f"DEBUG: Processing metrics2, tp length: {len(self.metrics2.stats['tp'])}")
                # LOGGER.info(f"DEBUG: metrics2 stats keys: {self.metrics2.stats.keys()}")
                # if len(self.metrics2.stats['tp']) > 0:
                # LOGGER.info(f"DEBUG: First tp shape: {self.metrics2.stats['tp'][0].shape}")
                # Don't plot during final evaluation to avoid file access issues
                # Just compute the metrics without saving plots
                self.metrics2.process(save_dir=self.save_dir / "cls2", plot=False, on_plot=self.on_plot)
                stats2 = self.metrics2.results_dict

                # Get cls2 fitness
                fitness2 = stats2.get("fitness", 0.0)

                # Add cls2 metrics with (M) suffix for "Material"
                for key, value in stats2.items():
                    new_key = key.replace("(B)", "(M)")  # Replace Box with Material
                    stats[new_key] = value

                # ⭐ Calculate combined fitness (weighted average)
                # Get mat_fitness_weight from trainer args or use default
                mat_weight = getattr(self.args, "mat_fitness_weight", 0.5)  # Default 0.5
                combined_fitness = (1 - mat_weight) * fitness1 + mat_weight * fitness2

                # Override original fitness with combined fitness
                stats["fitness"] = combined_fitness
                stats["fitness/cls1"] = fitness1  # Keep original cls1 fitness
                stats["fitness/cls2"] = fitness2  # Keep original cls2 fitness

                LOGGER.info(
                    f"Combined Fitness: {combined_fitness:.5f} (cls1: {fitness1:.5f} × {1 - mat_weight:.2f} + cls2: {fitness2:.5f} × {mat_weight:.2f})"
                )

                # Don't clear stats here - keep them for potential later access
                # self.metrics2.clear_stats()  # Commented out to preserve cls2 stats
            except Exception as e:
                LOGGER.warning(
                    f"Failed to process metrics2: {e}. This may be because cls2 labels are not available in the validation set."
                )
        elif self.metrics2 is not None:
            LOGGER.warning(
                "No cls2 data collected during validation. Check if cls2 labels are present in your dataset."
            )
            LOGGER.info(f"DEBUG: metrics2.stats: {self.metrics2.stats}")
            LOGGER.info(f"DEBUG: metrics2.stats['tp'] length: {len(self.metrics2.stats.get('tp', []))}")

        self.metrics.clear_stats()
        return stats

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        # Print primary classification (shape) results
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info("\n" + "=" * 100)
        LOGGER.info("Primary Classification (Shape) Results:")
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

        # Print secondary classification (material) results if available
        # Check if metrics2 has valid data (nt_per_class should exist and have data)
        if (
            self.metrics2 is not None
            and hasattr(self.metrics2, "nt_per_class")
            and self.metrics2.nt_per_class is not None
            and len(self.metrics2.nt_per_class) > 0
        ):
            LOGGER.info("\n" + "=" * 100)
            LOGGER.info("Secondary Classification (Material) Results:")
            pf2 = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics2.keys)
            LOGGER.info(pf2 % ("all", self.seen, self.metrics2.nt_per_class.sum(), *self.metrics2.mean_results()))

            # Print per-class results only during final validation (not during training)
            # During training: only show summary; Final validation: show all details
            if not self.training and self.nc2 > 1 and len(self.metrics2.ap_class_index):
                for i, c in enumerate(self.metrics2.ap_class_index):
                    LOGGER.info(
                        pf2
                        % (
                            self.names2[c],
                            self.metrics2.nt_per_image[c],
                            self.metrics2.nt_per_class[c],
                            *self.metrics2.class_result(i),
                        )
                    )
            LOGGER.info("=" * 100 + "\n")
        elif self.metrics2 is not None and self.nc2 > 0:
            LOGGER.info("\n" + "=" * 100)
            LOGGER.info("Secondary Classification (Material) Results:")
            LOGGER.info("⚠️  No material classification data collected during validation")
            LOGGER.info("   This may be because cls2 labels are not present in the validation set")
            LOGGER.info("=" * 100 + "\n")

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Return correct prediction matrix.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset, batch_size, self.args.workers, shuffle=False, rank=-1, drop_last=self.args.compile
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            names2=self.names2 if hasattr(self, "names2") else None,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        # TODO: optimize this
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        # TODO: fix this
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            names2=self.names2 if hasattr(self, "names2") else None,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """
        Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics
        for object detection. Updates the provided stats dictionary with computed metrics
        including mAP50, mAP50-95, and LVIS-specific metrics if applicable.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path]): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path]): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | list[str]]): IoU type(s) for evaluation. Can be single string or list of strings.
                Common values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | list[str]]): Suffix to append to metric names in stats dictionary. Should correspond
                to iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
