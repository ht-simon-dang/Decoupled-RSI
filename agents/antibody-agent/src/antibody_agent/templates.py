from .contracts import Brief, Product

TEMPLATES = {
    "disease": {"name": "疾病机制探索", "description": "建立作用方向、靶点状态、功能表位和格式的待决档案"},
    "lead": {"name": "配对先导优化", "description": "填入已有 VH/VL、可信复合物与明确的 IMGT 可编辑位点"},
    "vhh": {"name": "VHH 独立评价", "description": "提供单域候选与抗原状态，使用对应的结构和开发基准"},
}


def template(name):
    if name == "disease":
        return Brief(name="疾病导向抗体研究").model_dump()
    if name == "lead":
        return Brief(
            name="配对先导优化", entry="complex", route="optimize", generators=["antifold"]
        ).model_dump()
    if name == "vhh":
        return Brief(
            name="VHH 候选分析",
            entry="sequence",
            route="library",
            generators=["library"],
            product=Product(
                format="VHH", valency=1, fc="none", fc_requirement="none", chain_organization="single VHH"
            ),
        ).model_dump()
    raise KeyError("Unknown template")
