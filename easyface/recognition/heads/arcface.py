from .arcmargin import ArcMargin


class ArcFace(ArcMargin):
    def __init__(self, in_feats, out_feats, s=64.0, m=0.5) -> None:
        super().__init__(in_feats, out_feats, s=s, m2=m)