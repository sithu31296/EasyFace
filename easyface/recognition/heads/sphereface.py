from .arcmargin import ArcMargin


class SphereFace(ArcMargin):
    def __init__(self, in_feats, out_feats, m=4) -> None:
        super().__init__(in_feats, out_feats, s=1, m1=m)