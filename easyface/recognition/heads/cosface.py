from .arcmargin import ArcMargin


class CosFace(ArcMargin):
    def __init__(self, in_feats, out_feats, s=1, m=0.35) -> None:
        super().__init__(in_feats, out_feats, s=s, m3=m)