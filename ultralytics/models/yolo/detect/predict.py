# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        # Determine number of class channels to pass to NMS. By default NMS tries to infer nc from prediction shape
        # but when a model has a secondary-class head (nc2) we must include those channels so they aren't treated as
        # 'extra' mask data. Compute primary nc from model.names or model.nc and add nc2 if present.
        try:
            nc_primary = len(getattr(self.model, "names", [])) if getattr(self.model, "names", None) is not None else int(getattr(self.model, "nc", 0))
        except Exception:
            nc_primary = 0
        nc2 = int(getattr(self.model, "nc2", 0) or 0)
        nc_for_nms = nc_primary + nc2 if self.args.task == "detect" else nc_primary

        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=nc_for_nms,
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def get_obj_feats(self, feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # If model outputs include secondary-class probabilities (cls2), they will be appended after cls in raw preds
        # Results expects boxes in shape (N,6) for xyxy,conf,cls; we keep that and also pass cls2 if present
        boxes = pred[:, :6]
        cls2 = None
        cls2_probs = None
        # Determine extra columns (if any) and whether model has a secondary class head (nc2)
        extra = pred.shape[1] - 6 if pred.ndim == 2 else 0
        # try to find nc2 and names2 from model internals or YAML
        nc2 = None
        names2 = getattr(self.model, "names2", None)
        try:
            inner = getattr(self.model, "model", None)
            if inner is not None:
                # inner may be DetectionModel instance
                det_head = inner.model[-1] if hasattr(inner, 'model') else inner[-1]
                nc2 = getattr(det_head, "nc2", None)
                if names2 is None:
                    names2 = getattr(inner, "names2", None)
        except Exception:
            pass

        # fallback: try to read from AutoBackend.yaml if available
        if nc2 is None:
            try:
                yaml = getattr(self.model, 'yaml', None)
                if yaml and isinstance(yaml, dict):
                    nc2 = int(yaml.get('nc2', 0)) if yaml.get('nc2', None) is not None else None
                    if names2 is None:
                        names2 = yaml.get('names2', None)
            except Exception:
                pass

        # If names2 still None or appears to be numeric placeholders, try loading from predictor data or data yaml
        if names2 is None:
            try:
                data_cfg = getattr(self, 'data', None)
                if isinstance(data_cfg, dict):
                    names2 = data_cfg.get('names2', None)
            except Exception:
                names2 = None

        if names2 is None:
            try:
                data_arg = getattr(self.args, 'data', None)
                from pathlib import Path

                if isinstance(data_arg, str) and Path(data_arg).exists():
                    try:
                        import yaml as _yaml

                        y = _yaml.safe_load(Path(data_arg).read_text())
                        names2 = y.get('names2', None)
                    except Exception:
                        names2 = None
            except Exception:
                names2 = None

        # Repository-level fallback: search datasets/*/data.yaml for names2 mappings
        if names2 is None:
            try:
                from pathlib import Path
                import yaml as _yaml

                # find repo root and look for a top-level datasets/ directory
                repo_datasets = Path(__file__).resolve().parents[5] / 'datasets'
                if repo_datasets.exists():
                    for p in repo_datasets.rglob('data.yaml'):
                        try:
                            y = _yaml.safe_load(p.read_text())
                            if y and isinstance(y, dict) and y.get('names2', None) is not None:
                                names2 = y.get('names2')
                                break
                        except Exception:
                            continue
            except Exception:
                names2 = None

        # Normalize names2 to a dict[int,str] if it's a list or has string keys
        if names2 is not None:
            try:
                # If names2 is a list -> convert to dict
                if isinstance(names2, (list, tuple)):
                    names2 = {int(i): str(n) for i, n in enumerate(names2)}
                elif isinstance(names2, dict):
                    # ensure keys are ints and values are strings
                    normalized = {}
                    for k, v in names2.items():
                        try:
                            ik = int(k)
                        except Exception:
                            ik = len(normalized)
                        normalized[ik] = str(v)
                    names2 = normalized
                else:
                    # unexpected type -> ignore
                    names2 = None
            except Exception:
                names2 = None

        # If names2 exists but looks like numeric placeholders (values equals their keys or digits), try to replace
        # with dataset YAML mapping so we plot human-readable secondary-class names.
        def _looks_like_numeric_placeholders(m):
            if not isinstance(m, dict) or not m:
                return False
            try:
                for k, v in m.items():
                    if str(v).strip() != str(k).strip() and not str(v).strip().isdigit():
                        return False
                return True
            except Exception:
                return False

        if names2 is not None and _looks_like_numeric_placeholders(names2):
            # try to find a better names2 from dataset YAMLs
            try:
                from pathlib import Path
                import yaml as _yaml

                repo_root = Path(__file__).resolve().parents[4]
                repo_datasets = repo_root / 'datasets'
                if repo_datasets.exists():
                    for p in repo_datasets.rglob('data.yaml'):
                        try:
                            y = _yaml.safe_load(p.read_text())
                            cand = y.get('names2', None) if isinstance(y, dict) else None
                            if cand:
                                # normalize candidate
                                if isinstance(cand, (list, tuple)):
                                    cand = {int(i): str(n) for i, n in enumerate(cand)}
                                elif isinstance(cand, dict):
                                    nc = {}
                                    for kk, vv in cand.items():
                                        try:
                                            ik = int(kk)
                                        except Exception:
                                            ik = len(nc)
                                        nc[ik] = str(vv)
                                    cand = nc
                                else:
                                    cand = None
                                if cand:
                                    names2 = cand
                                    break
                        except Exception:
                            continue
            except Exception:
                pass

        # Extract cls2 depending on extra shape and nc2
        if extra > 0 and names2 is not None:
            try:
                if nc2 and extra == int(nc2):
                    # last `nc2` columns are secondary-class probabilities -> argmax
                    cls2_probs = pred[:, -extra:]
                    cls2 = cls2_probs.argmax(dim=1).long()
                elif extra == 1:
                    # last column is a single index per box
                    cls2 = pred[:, -1].long()
                else:
                    # unknown layout: try last column
                    cls2 = pred[:, -1].long()
            except Exception:
                cls2 = None

        return Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=boxes,
            names2=names2,
            cls2=cls2,
            cls2_probs=cls2_probs,
        )
