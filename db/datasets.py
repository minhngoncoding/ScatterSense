from db.coco import Chart
from db.coco import MSCOCO
from db.coco import SKU
from db.coco import Pie, Line, Bar, Cls, LineCls, LineClsReal
from db.coco import Scatter, Box

datasets = {
    "Chart": Chart,
    "MSCOCO": MSCOCO,
    "SKU110": SKU,
    "Pie": Pie,
    "Line": Line,
    "Bar": Bar,
    "Cls": Cls,
    "LineCls": LineCls,
    "LineClsReal": LineClsReal,
    "Scatter": Scatter,
    "Box": Box,
}
