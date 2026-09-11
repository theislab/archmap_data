"""Compatibility shims for the third-party versions this repo pins.

Importing this module applies the patches. It is idempotent, so it is safe to
import from every entrypoint that touches scarches/scvi-tools.
"""

from collections.abc import Mapping

import anndata


def patch_anndata_layers_none():
    """Let ``AnnData(X=..., layers=...)`` tolerate a ``None`` key in ``layers``.

    anndata 0.13 made ``.X`` addressable as ``layers[None]``, so iterating
    ``adata.layers`` now yields ``None`` alongside the real layer names. Both
    scarches and scvi-tools build a zero-padding AnnData with

        AnnData(X=pad.copy(), layers={layer: pad.copy() for layer in adata.layers})

    which under anndata >=0.13 puts a ``None`` key in ``layers`` holding a
    *different object* than the ``X`` passed alongside it. anndata compares the
    two by identity and raises

        ValueError: If you provide `layers[None]` and `X`, they must be identical.

    Affected call sites, all reached by the mapping pipeline:
      * ``scvi.model.base._archesmixin._pad_and_sort_query_anndata`` -- runs
        inside ``prepare_query_anndata()`` whenever the query is missing genes
        present in the reference, i.e. for essentially every user-supplied
        query. Fixed upstream in scvi-tools 1.4.3; we pin 1.4.2.
      * ``scvi.model.utils._minification._get_minified_adata_scrna``
      * ``scarches.models.base._utils.get_minified_adata_scrna``
        Both still unfixed upstream as of scvi-tools 1.5.1.

    Dropping the ``None`` entry when an explicit ``X`` was passed restores the
    pre-0.13 behaviour exactly: ``X`` becomes ``.X`` (and hence ``layers[None]``)
    and only the real layers are carried over.
    """
    if getattr(anndata, "_archmap_layers_none_shim", False):
        return

    _orig_init = anndata.AnnData.__init__

    def _anndata_init_compat(self, *args, **kwargs):
        layers = kwargs.get("layers")
        X = args[0] if args else kwargs.get("X")
        if (
            X is not None
            and not isinstance(X, anndata.AnnData)
            and isinstance(layers, Mapping)
            and None in layers
        ):
            kwargs["layers"] = {k: v for k, v in layers.items() if k is not None}
        return _orig_init(self, *args, **kwargs)

    anndata.AnnData.__init__ = _anndata_init_compat
    anndata._archmap_layers_none_shim = True


patch_anndata_layers_none()
