# MoDora 代码级调研报告

> 调研对象：`MoDora` 官方 GitHub 仓库。
> 调研口径：以下内容严格基于当前仓库中的后端代码、prompt 与 README，而不是仅根据论文描述推断。MoDora 官方 README 将其定义为面向半结构化文档分析的 LLM\-powered framework，核心思想是用 Component\-Correlation Tree，即 CCTree，建模复杂版式、多元素半结构化文档，并支持文档管理、结构可视化、热力图和多文档分析等功能。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/README.md)\)
> 本报告飞书链接：[飞书](https://ucn5mshih8o6.feishu.cn/wiki/YZxRwz2kliaZtbkVkPdchbKjnMc)
> 
> 

---

## 1. 总体流程概览

从代码看，MoDora 可以分成两条主流程：

```Plain Text
一、文档预处理 / 入库 / 建树流程

PDF
  → OCR 或 PDF text fallback
  → OCRBlock 列表
  → StructureAnalyzer
  → ComponentPack
      ├─ body: text / image / chart / table 等正文组件
      └─ supplement: header / footer / number / aside_text 等辅助信息
  → EnrichmentService
      └─ 对 image / chart / table 做截图 + LLM 描述增强
  → AsyncLevelGenerator
      └─ 用 LLM 判断标题层级，生成 title_level
  → TreeConstructor
      └─ 构建 Component-Correlation Tree
  → AsyncMetadataGenerator
      └─ 自底向上生成 / 整合 metadata keywords
  → tree.json / cp.json / ocr.json / knowledge_base.json


二、查询问答流程

Query
  → extract_location(query)
      ├─ 能抽取页码 / 页面位置 → LocationRetriever
      └─ 无明确位置线索 → SemanticRetriever + optional VectorRetriever
  → RetrievalResult
      ├─ text_map
      ├─ locations
      ├─ locations_by_path
      └─ locations_by_file_page
  → crop evidence images
  → LLM reason_retrieved()
  → check_answer()
      ├─ 验证通过 → 返回答案
      └─ 验证失败 → 单文档 whole-document fallback
```

仓库 README 中的 CLI 命令也对应这条主线：`ocr` 负责 OCR 与组件抽取，`build-tree` 负责从组件构建 `tree.json`，`qa`负责单文档问答，`batch-qa` 负责批量实验。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/README.md)\)

---

## 2. 核心数据结构

### 2\.1 `OCRBlock`

`OCRBlock` 是 OCR 阶段输出的最低级结构化单元。它包含：

```Python
page_id: int
block_id: int
bbox: list[float]
label: str
content: str
```

其中 `bbox=[x0, y0, x1, y1]` 是该 OCR 块在 PDF 页面坐标系中的位置；`label` 用于区分标题、正文、图片、图表、表格、页眉、页脚、页码、侧栏文字等类型。代码中通过 `is_title()`、`is_figure()`、`is_figure_title()`、`is_header()`、`is_footer()`、`is_number()`、`is_aside()` 等方法判断 OCR block 的类别。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/ocr.py)\)

可以理解为：

```Plain Text
OCRBlock = 页面上的一个低级 OCR 识别块
```

它还不是 MoDora 真正用于问答的语义组件。

---

### 2\.2 `Location`

`Location` 保存组件在 PDF 中的位置：

```Python
bbox: list[float]
page: int
file_name: str | None = None
```

其中 `file_name` 主要用于多文档场景。也就是说，MoDora 后续每个组件都可以追溯到 PDF 中的某个页面、某个 bbox 区域；这也是后面 location retrieval、证据截图和前端高亮的基础。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/component.py)\)

---

### 2\.3 `Component`

`Component` 是 OCRBlock 进一步聚合后的语义组件。代码定义中，`Component` 包含：

```Python
type: str
title: str
title_level: int = 1
metadata: Any | None = None
data: str = ""
location: list[Location]
```

其中：

代码注释称 `Component` 是从 PDF 中抽取出的最小语义单元，例如一段文本、一张图片、一张表格等。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/component.py)\)

---

### 2\.4 `Supplement`

`Supplement` 专门保存辅助页面信息，包括：

```Python
header: dict[int, Component]
footer: dict[int, Component]
number: dict[int, Component]
aside: dict[int, Component]
```

这些信息不是直接混入正文组件，而是按页号聚合。这样做的好处是：正文内容和页眉、页脚、页码、侧栏文字可以分开组织，避免常规语义问答时被辅助信息干扰。\(GitHub\)

---

### 2\.5 `ComponentPack`

`ComponentPack` 是一个文档级组件包，包含：

```Python
body: list[Component]
supplement: Supplement
```

其中 `body` 是文档主体组件列表，`supplement` 是页眉、页脚、页码、侧栏等辅助组件集合。代码注释明确说明，`ComponentPack` 用于保存整个文档的组件集合。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/component.py)\)

需要特别说明：`co_pack.body` 不是一个单独的大组件，而是一个 **组件列表**。当代码说把某个组件 flush 到 `co_pack.body` 时，本质是把该组件 append / extend 到这个列表中，不是把多个组件合并成一个组件。

---

### 2\.6 `CCTreeNode` 与 `CCTree`

`CCTreeNode` 是 Component\-Correlation Tree 的节点。它包含：

```Python
type
metadata
data
location
children
height
depth
keyword_cnt
impact
```

相比普通 RAG 中的 chunk，CCTree 节点不仅有文本 `data`，还保存组件类型、metadata、PDF 位置、树结构 children、节点高度深度、关键词数量和 impact 统计。代码中 `CCTreeNode.from_component()` 会把 `Component` 转换成树节点。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/cctree.py)\)

`CCTree` 则是整棵文档树的封装，核心字段是：

```Python
root: CCTreeNode
```

它还提供 `get_structure()`、`get_clean_structure()`、`find_node_by_path()`、`update_impact()`、`merge_multi_trees()` 等方法。其中 `get_structure()` 默认排除 `Supplement` 节点，只保留树骨架；`get_clean_structure()` 会保留节点 data，用于 fallback 问答。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/cctree.py)\)

---

### 2\.7 `RetrievalResult`

`RetrievalResult` 是检索阶段的结果对象，包含：

```Python
text_map: Dict[str, str]
locations: List[Location]
locations_by_path: Dict[str, List[Location]]
locations_by_file_page: Dict[tuple[str | None, int], List[Location]]
```

这几个变量的区别非常关键：

`normalize_locations()` 会对位置做去重，并根据 `locations` 生成 `locations_by_file_page`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/cctree.py)\)

---

## 3. 文档处理入口：`process_document_task`

后端服务中的文档处理入口是 `process_document_task()`。它的流程是：

```Plain Text
1. 设置任务状态为 processing
2. 优先加载 OCR model
3. 如果 OCR model 不可用，fallback 到 PDF text extraction
4. 得到 OcrExtractResponse
5. 缓存 ocr.json
6. 调用 get_components_async() 生成 ComponentPack
7. 缓存 cp.json
8. 调用 build_tree_async() 生成 CCTree
9. 从 root metadata 中抽取 semantic_tags
10. 保存 tree.json
11. 更新 knowledge_base.json 中的统计信息
```

代码中明确写了：优先使用 OCR 模型；如果模型加载失败或 OCR 预测失败，则使用 PDF text fallback，以保证流程继续执行。之后会缓存 `ocr.json`、`cp.json`、`tree.json`，并更新知识库统计信息，如页面数、组件数量、layout variance、节点数、叶子数、树深度等。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/document_processing.py)\)

这说明 MoDora 的离线处理不是只生成一个向量库，而是生成了一组中间结构：

```Plain Text
ocr.json: OCR 原始结构化结果
cp.json: ComponentPack
tree.json: CCTree
knowledge_base.json: 文档统计信息和 semantic tags
```

---

## 4. OCRBlock → ComponentPack：结构聚合流程

### 4\.1 `StructureAnalyzer.analyze()`

`StructureAnalyzer` 的作用是把 flat OCR block list 转换成结构化的 `ComponentPack`。核心变量包括：

```Python
co_pack = ComponentPack()
cur_text_title = TITLE
cur_figure_title = TITLE
non_text_cache: list[Component] = []
cur_text_co = Component(type="text", title=cur_text_title)
```

它顺序遍历 OCR blocks，根据 block 类型分别处理。\(GitHub\)

---

### 4\.2 遇到标题块：flush 旧文本组件，开启新文本组件

当 `block.is_title()` 为真时，代码逻辑是：

```Plain Text
1. 读取当前标题文本 block.content
2. 如果当前文本组件已有内容，则把 cur_text_co append 到 co_pack.body
3. 把 non_text_cache 中缓存的非文本组件 extend 到 co_pack.body
4. 清空 non_text_cache
5. 用当前标题初始化新的 text component
6. 把标题自身 bbox 加入新 text component 的 location
7. 把标题文本写入新 text component 的 data
```

这一步中的 flush 不是“组件合并”，而是“提交组件到 body 列表”。`cur_text_co` 和 `non_text_cache` 中的 image/chart/table 仍然是不同组件；text 组件类型仍为 `text`，图片仍为 `image`，表格仍为 `table`，图表仍为 `chart`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/structure.py)\)

代码这样做的含义是：标题作为章节边界。一个标题出现时，说明上一个章节或段落组件已经结束，接下来开始新的文本组件。

---

### 4\.3 普通文本块：追加到当前 text component

如果当前 OCR block 不是标题、图片、图表、表格、页眉、页脚、页码、侧栏等特殊类型，则会进入默认分支：

```Plain Text
cur_text_co.data += "\n\n" + block.content
cur_text_co.location.append(location)
```

因此，一个 `text` Component 通常由多个 OCRBlock 聚合而来。聚合方式是：

```Plain Text
data: 文本内容拼接
location: 多个 bbox 位置追加
type: 仍然是 text
title: 当前章节标题
```

这才是代码层面的“组件聚合”：多个低级 OCR 文本块合并成一个更大的文本语义组件。\(GitHub\)

---

### 4\.4 图片 / 图表 / 表格：进入 `non_text_cache`

当 `block.is_figure()` 为真时，即 block 类型属于 `image/chart/table`，代码会创建一个非文本组件：

```Python
cur_figure_co = Component(
    type=block.label,
    title=cur_figure_title,
    location=[location]
)
```

然后根据相邻 OCR block 判断 figure title：

```Plain Text
如果前一个 block 是 figure_title / vision_footnote
  → 认为标题在图表上方

如果后一个 block 是 figure_title / vision_footnote
  → 认为标题在图表下方

否则
  → 使用 Default Title
```

创建出的非文本组件不会立即放入 `co_pack.body`，而是先放入 `non_text_cache`。等下一个标题出现、或者遍历结束时，再被 flush 到 `co_pack.body`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/structure.py)\)

这说明 MoDora 并不是把图片、表格、图表直接塞进当前 text component，而是让它们保持独立组件，再在建树阶段挂到当前章节节点下面。

---

### 4\.5 页眉、页脚、页码、侧栏：进入 `supplement`

对于 header、footer、number、aside\_text，代码不会把它们加入 `co_pack.body`，而是按页号聚合到：

```Python
co_pack.supplement.header
co_pack.supplement.footer
co_pack.supplement.number
co_pack.supplement.aside
```

如果同一页有多个 header/footer/number/aside block，则代码会把文本拼接到已有组件的 `data`，并把 bbox 追加到已有组件的 `location`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/structure.py)\)

所以这里的聚合方式是：

```Plain Text
同一页、同一辅助类型的 OCRBlock
  → 合并为一个 supplement Component
```

例如：

```Plain Text
第 2 页多个 footer block
  → supplement.footer[2]
```

---

## 5. EnrichmentService：非文本组件语义增强

### 5\.1 Enrichment 发生在组件聚合之后

`get_components_async()` 的顺序是：

```Plain Text
1. StructureAnalyzer.analyze()
2. EnrichmentService.enrich_async()
```

也就是说，先从 OCRBlock 得到 `ComponentPack`，再对其中的非文本组件做 enrichment。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/preprocess.py)\)

---

### 5\.2 Enrichment 只处理 image / chart / table

`EnrichmentService.enrich_async()` 会遍历 `co_pack.body`，只挑选：

```Python
if co.type in ["image", "chart", "table"]:
    tasks.append(co)
```

然后对这些组件执行：

```Plain Text
1. 根据组件 location 从 PDF 中裁剪图片
2. 调用 LLM 生成 annotation
3. 回填 title / metadata / data
```

如果组件本身 title 是默认标题，LLM 生成的 title 会覆盖它；metadata 会被写入组件；data 如果原来为空，则写入 LLM 生成的内容描述。\(GitHub\)

---

### 5\.3 三类 prompt：image / chart / table

Enrichment prompt 要求 LLM 固定输出三类属性：

```Plain Text
[T] title
[M] metadata
[C] content
```

image prompt 要求详细描述图片内容；chart prompt 强调图表标题、坐标轴、趋势和具体数值；table prompt 强调表格结构、列含义和表格内容。\(GitHub\)

因此，一个 chart 组件在 enrichment 后可能变成：

```Plain Text
type = chart
title = 图表标题
metadata = 图表主题、横纵轴、趋势关键词
data = 图表内容的详细自然语言描述
location = 图表 bbox
```

这一步对应论文中的 type\-specific information extraction，但代码实现更具体：它主要是对非文本组件做截图 \+ 多模态 LLM annotation。

---

## 6. 标题层级生成：`AsyncLevelGenerator`

在 `StructureAnalyzer` 刚结束时，text component 虽然有 `title`，但这些 title 之间基本没有可靠层级。`Component.title_level` 默认是 1；代码没有在 OCR 聚合阶段直接判断一级标题、二级标题、三级标题。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/component.py)\)

真正判断标题层级的是 `AsyncLevelGenerator.generate_level()`。它会：

```Plain Text
1. 从 cp.body 中筛选 type == "text" 的组件
2. 取这些组件的 title 列表
3. 取每个 text component 的第一个 location 作为标题区域
4. 从 PDF 中裁剪标题区域图像
5. 调用 LLM generate_levels(title_list, image)
6. 期望 LLM 返回 Markdown 风格标题列表
7. 通过统计 # 的数量得到 title_level
```

例如：

```Plain Text
"Introduction"       → "# Introduction"       → title_level = 1
"Related Work"      → "## Related Work"      → title_level = 2
"Vector Retrieval"  → "### Vector Retrieval" → title_level = 3
```

如果 LLM 返回的标题前没有 `#`，则 `_get_title_level()` 默认返回 1。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/hierarchy.py)\)

所以代码中的标题层级不是由 OCR label 直接给出的，而是由 LLM 结合标题文本和标题区域视觉信息生成。

---

## 7. ComponentPack → CCTree：`TreeConstructor`

### 7\.1 文本组件按 `title_level` 建树

`TreeConstructor.construct_tree()` 首先创建 root：

```Python
root = CCTreeNode(type="root", metadata="", data="", location=[], children={})
stack = [(root, -1)]
```

然后遍历 `cp.body`。如果当前组件是 `text`，就读取它的 `title_level`，并用栈维护层级关系：

```Plain Text
while stack 顶部层级 >= 当前标题层级:
    pop

parent = 当前 stack 顶部节点
parent.children[component.title] = 当前 text node
当前 text node 入栈
```

这类似 Markdown 标题建树逻辑：

```Plain Text
# A
  ## A.1
  ## A.2
# B
  ## B.1
```

在树里会变成：

```Plain Text
root
├─ A
│  ├─ A.1
│  └─ A.2
└─ B
   └─ B.1
```

代码中还用 `_uniq_key()` 处理同一个父节点下重复标题的问题，避免 children key 冲突。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/constructor.py)\)

---

### 7\.2 非文本组件挂到当前章节下

如果组件类型是：

```Python
{"image", "table", "chart"}
```

代码不会再根据它们自己的 title\_level 建树，而是直接挂到当前 stack top，也就是最近的文本章节节点下面。\(GitHub\)

这意味着：

```Plain Text
某章节下出现的图片 / 表格 / 图表
  → 作为该章节的子节点
```

这也是 CCTree 表达“组件相关性”的一个关键点：非文本组件通过其出现位置和当前章节上下文，与章节节点建立父子关系。

---

### 7\.3 Supplement 被挂成独立辅助子树

`cp.supplement` 会被构造成：

```Plain Text
root
└─ Supplement
   ├─ header
   │  ├─ Header of Page 1
   │  └─ Header of Page 2
   ├─ footer
   ├─ number
   └─ aside
```

每个 supplement component 会被转换成 CCTreeNode，metadata 和 data 都设置为其文本内容。`get_structure()` 默认会排除 key 为 `"Supplement"` 的节点，因此常规问答 schema 中不会展示辅助信息。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/constructor.py)\)

---

## 8. Bottom\-up metadata generation

`AsyncMetadataGenerator` 负责为 CCTree 节点生成 semantic metadata。代码注释说它生成的是 semantic keywords。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/generator.py)\)

流程是 DFS：

```Plain Text
1. 先递归处理子节点
2. 更新当前节点 depth / height
3. 根据子节点 keyword_cnt、当前 depth、height 计算当前节点 keyword_cnt
4. 对当前节点生成或整合 metadata
```

对于不同类型节点：

```Plain Text
text node:
  先基于 node.data 生成 metadata
  再整合子节点 metadata

root node:
  只整合子节点 metadata

其他节点:
  如果 metadata 为空，则默认用 node.data 作为 metadata
```

关键词数量计算使用 `n0`、`growth_rate`、子节点 keyword 总数、节点 depth、height 等因素。代码默认在 `build_tree_async()` 中构造：

```Python
AsyncMetadataGenerator(n0=2, growth_rate=2.0, ...)
```

metadata prompt 要求 LLM 生成若干名词短语关键词，使用分号分隔；integration prompt 则要求从一组关键词中选择或总结出更综合的关键词。\(GitHub\)

因此，论文里的 bottom\-up cascade summarization，在当前代码中更准确地说是：

```Plain Text
自底向上的层级关键词生成与整合
```

它不是生成长段自然语言摘要，而是生成 / 整合 semantic keywords。

---

## 9. 查询问答流程：`QAService.qa()`

### 9\.1 先抽取 location cues

`QAService.qa()` 的第一步是：

```Python
page_list, position_vector = await self.extract_location(query)
```

`extract_location()` 使用 `location_extraction_prompt`，让 LLM 从自然语言问题里抽取：

```Plain Text
Page: [page_numbers]; Position: [row, column]
```

其中，页面被划分为 3×3 网格：

```Plain Text
row:
top    = 1
middle = 2
bottom = 3
未提及 = -1

column:
left   = 1
center = 2
right  = 3
未提及 = -1
```

如果问题没有页码，返回 `[-1]`；如果没有页面位置，返回 `[-1, -1]`。Prompt 中的例子包括：“第一页右上角写了什么” → `Page: [1]; Position: [1, 3]`；“第 6 页底部中间写了什么” → `Page: [6]; Position: [3, 2]`；“文档标题是什么” → `Page: [-1]; Position: [-1, -1]`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/prompts/retrieval.py)\)

所以用户并不需要显式输入“位置向量”。位置向量是 LLM 从“top right / bottom center / first page / page 6”等自然语言中解析出来的。

---

### 9\.2 检索路由逻辑

`QAService.qa()` 的路由逻辑是：

```Python
if -1 in page_list and position_vector == [-1.0, -1.0]:
    semantic_result = await self.semantic_retriever.retrieve(...)
    if self.vector_retriever:
        vector_result = await self.vector_retriever.retrieve(...)
    result.update(semantic_result)
    result.update(vector_result)
else:
    result = self.location_retriever.retrieve(...)
```

也就是说：

```Plain Text
没有明确页码 + 没有明确页面位置
  → SemanticRetriever + optional VectorRetriever

有页码或页面位置线索
  → LocationRetriever
```

需要注意：代码里不是先调用一个显式的“问题类型分类器”，而是通过 `extract_location()` 的结果来决定检索路径。虽然仓库里存在更复杂的 `question_parsing_prompt`，但当前 `QAService.qa()` 使用的是 `location_extraction_prompt`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

---

## 10. LocationRetriever：基于 3×3 网格和 bbox overlap 的版式检索

### 10\.1 页码解析

`LocationRetriever.retrieve()` 首先打开 PDF，获取页数。如果 `page_list` 中包含 `-1`，则 `_resolve_page_list()` 会把目标页扩展为所有页面；否则只检索指定页。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/retrieve/location_retriever.py)\)

例如：

```Plain Text
page_list = [2]
  → 只检索第 2 页

page_list = [-1]
  → 检索全部页面
```

---

### 10\.2 bbox 归一化

每个组件的 `Location.bbox` 是 PDF 页面上的绝对坐标：

```Plain Text
bbox = [x0, y0, x1, y1]
```

但是不同 PDF 页面的宽高不同，所以代码先把 bbox 归一化到 `[0, 1]`：

```Python
[
  bbox[0] / page_width,
  bbox[1] / page_height,
  bbox[2] / page_width,
  bbox[3] / page_height,
]
```

这样一来，所有页面都可以统一用 3×3 网格表示。\(GitHub\)

---

### 10\.3 3×3 网格区域

如果 `position_vector=[1,3]`，表示 top right：

```Plain Text
row = 1
column = 3
```

对应的归一化区域是：

```Plain Text
x: [(3-1)/3, 3/3] = [0.666..., 1.0]
y: [(1-1)/3, 1/3] = [0.0, 0.333...]
```

如果 row 或 column 是 `-1`，表示该方向不限制：

```Plain Text
[1, -1]  → 顶部整行
[-1, 3]  → 右侧整列
[-1, -1] → 整页
```

代码中 grid 区域就是这样计算的。\(GitHub\)

---

### 10\.4 overlap 判断

代码通过如下逻辑判断组件 bbox 是否与目标网格重叠：

```Python
x_overlap = not (x1 <= grid_x0 or x0 >= grid_x1)
y_overlap = not (y1 <= grid_y0 or y0 >= grid_y1)
return x_overlap and y_overlap
```

也就是说，只要组件 bbox 和目标网格在 x、y 两个方向都存在交集，就认为该组件命中。命中后：

```Plain Text
如果 node.data 非空:
  result.text_map[path] = node.data

result.locations.extend(hit_locations)
result.locations_by_path[path].extend(hit_locations)
```

最后会给 location 补充 `file_name`，并调用 `normalize_locations()` 做去重和按文件页聚合。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/retrieve/location_retriever.py)\)

---

## 11. SemanticRetriever：LLM\-guided pruning

### 11\.1 基本流程

`SemanticRetriever.retrieve()` 从：

```Python
nodes = {"root": tree.root}
```

开始递归处理树节点。每一层会：

```Plain Text
1. 并发处理当前层节点
2. 对每个节点判断是否相关
3. 如果相关，收集该节点 data 和 location
4. 如果该节点有 children，则让 LLM 选择下一层相关子节点
5. 递归进入被选中的子节点
```

当前层处理完后，`_retrieve_recursive()` 会继续处理下一层 selected children。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py)\)

---

### 11\.2 相关性判断：`_is_relevant()`

对于单个节点，`_process_single_node()` 会先判断：

```Python
if node.has_content() and await self._is_relevant(...):
```

`node.has_content()` 的定义是：

```Python
return bool(self.data or self.location)
```

如果节点有内容，则 `_is_relevant()` 会：

```Plain Text
1. 拼接当前路径末尾标题和 node.data
2. 根据 node.location 从 PDF 裁剪图像
3. 调用 LLM check_node_mm()
4. 判断图像或文本是否包含回答 query 的证据或线索
```

`check_node_mm` 使用的 prompt 要求模型只返回 `T` 或 `F`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/cctree.py)\)

---

### 11\.3 LLM 看到的是直接子节点，不是所有子孙节点

当一个节点被判断相关且有 children 时，代码调用 `_select_children()`。它传给 LLM 的是：

```Python
children_list = list(node.children.keys())
metadata_map = node.get_metadata_map()
```

而 `get_metadata_map()` 返回的是当前节点 **直接子节点** 的 `{child_key: child.metadata}`。因此，LLM\-guided pruning 中，LLM 每次看到的是当前节点的一层直接子节点，而不是所有子孙后代。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/cctree.py)\)

---

### 11\.4 多个相关子节点会同时进入下一层

`select_children_prompt` 要求 LLM 从候选标题列表中返回一个 list。这个 list 可以是输入列表本身、子集或空列表。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/prompts/retrieval.py)\)

如果 LLM 返回多个标题，例如：

```Python
["Section 2", "Section 4", "Table 1"]
```

`_get_children()` 会逐个检查这些 title 是否在当前 node\.children 中；存在则加入 `selected_children`。下一轮递归会同时处理这些子节点。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/retrieve/semantic_retriever.py)\)

因此，它不是单路径搜索，而是多分支剪枝搜索。

---

## 12. VectorRetriever：可选向量检索 pipeline

### 12\.1 何时启用

`QAService.init()` 中，只有当配置 `enable_vector_search` 为真时，才会构造 `VectorRetriever`。README 也说明，如果启用 vector search，需要配置 embedding API key；如果使用 rerank 模型，还需要 rerank API key。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

---

### 12\.2 VectorRetriever 的内部流程

`VectorRetriever.retrieve()` 的流程是：

```Plain Text
1. 遍历 CCTree，收集所有有 data 且 has_content() 的节点
2. 对每个节点构造检索文本
   path title + metadata + data
3. 生成 doc_id = source::path
4. 计算文本 hash
5. 写入或更新 Chroma collection
6. 对 query 生成 embedding
7. 在 Chroma 中查询 top_k * 4 个候选
8. 用 cosine distance 转成 score = 1 - dist
9. 过滤低于 min_score 的候选
10. 用 rerank client 重排
11. 取 top_k
12. 把候选节点写入 RetrievalResult
```

节点文本构造函数 `_build_node_text()` 会把路径末尾标题、metadata、data 拼接起来。最终命中的节点会写入 `result.text_map[path]`，其 location 会写入 `result.locations` 和 `result.locations_by_path`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/retrieve/vector_retriever.py)\)

---

### 12\.3 SemanticRetriever 与 VectorRetriever 是两条独立 pipeline

在没有明确位置线索时，`QAService.qa()` 会先跑 SemanticRetriever，再在启用向量检索的情况下跑 VectorRetriever，最后把两者结果合并：

```Python
result.update(semantic_result)
result.update(vector_result)
```

这个合并不是复杂的排序融合，而是 `RetrievalResult.update()` 的直接合并：

```Plain Text
text_map.update(other.text_map)
locations.extend(other.locations)
locations_by_path 逐 path 合并
normalize_locations()
```

`normalize_locations()` 会去重并重新生成 `locations_by_file_page`。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

所以更准确地说：

```Plain Text
SemanticRetriever 和 VectorRetriever 是两个相对独立的召回通道；
它们不共享中间推理过程；
只是在最后把 RetrievalResult union 到一起。
```

---

## 13. Reasoning、验证与 fallback

### 13\.1 正常问答 reasoning

检索完成后，`QAService.qa()` 会：

```Plain Text
1. 取 tree.get_structure() 作为 schema
2. 取 result.text_map 作为 textual evidence
3. 根据 result.locations 从 PDF 中裁剪证据图片
4. 调用 remote_llm.reason_retrieved()
```

`reason_retrieved()` 使用的 prompt 要求 LLM 基于 query、document schema、retrieved textual evidence 和对应视觉区域图片，返回简短答案。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

---

### 13\.2 `check_answer()` 并不验证事实正确性

生成答案后，代码会调用：

```Python
await self.remote_llm.check_answer(query, answer)
```

`check_answer_prompt` 的规则是：

```Plain Text
如果回答含义类似：
- no relevant information
- insufficient evidence
- unable to answer
- None
- N/A

则输出 F；
否则输出 T。
```

因此，这个验证更像是判断“回答是否有效 / 是否拒答 / 是否空答案”，不是判断答案是否真的正确，也不是与 reference answer 做语义对比。\(GitHub\)

代码中的 `_bool_string()` 也比较宽松：只要 LLM 返回字符串里包含 `t`、`yes` 或 `true`，就会被解析为 True。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/infra/llm/base.py)\)

---

### 13\.3 fallback：单文档 whole\-document reasoning

如果 `check_answer()` 返回 False，则代码会进入 fallback：

```Plain Text
如果 source_path 是单文档 str:
  1. 把整个 PDF 转成 base64 页面图像
  2. 取 tree.get_clean_structure()
  3. 调用 reason_whole()

如果 source_path 是 dict，即多文档:
  记录 warning：multi-doc mode does not support whole-page fallback yet
```

也就是说，whole\-document fallback 目前只支持单文档场景；多文档场景下没有完整的 whole\-page fallback。\(GitHub\)

---

## 14. `retrieved_documents` 与 bbox 聚合

最终返回前，`QAService.qa()` 会调用 `_format_retrieved_docs()`，把 `RetrievalResult` 格式化成 `retrieved_documents`。代码注释说明：它会把同一文件、同一页的 bboxes 组织到一个 evidence entry 中，并做 bbox 去重。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

也就是说，如果检索命中了：

```Plain Text
A.pdf 第 2 页 bbox1
A.pdf 第 2 页 bbox2
A.pdf 第 5 页 bbox3
B.pdf 第 1 页 bbox4
```

最终会组织成：

```JSON
[
  {
    "file_name": "A.pdf",
    "page": 2,
    "content": "...",
    "bboxes": [bbox1, bbox2],
    "retrievers": ["semantic", "vector"]
  },
  {
    "file_name": "A.pdf",
    "page": 5,
    "content": "...",
    "bboxes": [bbox3],
    "retrievers": ["semantic"]
  },
  {
    "file_name": "B.pdf",
    "page": 1,
    "content": "...",
    "bboxes": [bbox4],
    "retrievers": ["vector"]
  }
]
```

这里的“按文件名和页码聚合 bbox”就是：不是每个 bbox 返回一条 evidence，而是把同一文件同一页的多个 bbox 放进同一个 evidence entry 中，方便前端一次性高亮该页上的多个证据区域。\(GitHub\)

---

## 15. 多文档支持情况

### 15\.1 代码中有多文档树合并逻辑

`CCTree.merge_multi_trees(trees)` 可以把多个文档的 CCTree 合并到一个新的 `multi_doc_root` 下：

```Plain Text
multi_doc_root
├─ file_a.pdf
│  └─ file_a 的原始 CCTree root
├─ file_b.pdf
│  └─ file_b 的原始 CCTree root
└─ ...
```

在合并时，代码还会递归给每个 Location 注入 `file_name`，用于多文档检索和证据组织。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/domain/cctree.py)\)

---

### 15\.2 QAService 接收单文档或多文档输入

`QAService.qa()` 的 `source_path` 类型是：

```Python
str | dict[str, str]
```

这说明代码层面既支持单文档路径，也支持多文档 `{file_name: pdf_path}` 映射。SemanticRetriever 和 VectorRetriever 内部也都有处理 `source_path` 为 dict 的逻辑，例如根据路径中的文件名选择对应 PDF，或给 location 补充 file\_name。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

---

### 15\.3 但多文档支持不是所有模块都完整支持

代码中明确写到：

```Plain Text
Location retrieval currently only supports single documents
```

当走 location retrieval 且 `source_path` 是 dict 时，代码只取 `source_path.values()` 中的第一个文件作为 actual\_source。\([GitHub](https://raw.githubusercontent.com/OpenDataBox/MoDora/main/MoDora-backend/src/modora/core/services/qa_service.py)\)

另外，多文档场景下 whole\-page fallback 也不支持，代码只记录 warning。\(GitHub\)

因此，MoDora 的多文档支持更准确地说是：

```Plain Text
支持：
- 多文档管理
- 多文档 CCTree 合并
- 多文档 semantic / vector 检索结果组织

限制：
- layout-based location retrieval 当前主要是单文档实现
- whole-document fallback 当前不支持多文档
- SemanticRetriever 还存在 root 空节点可能阻断递归的代码 caveat
```

README 中也写到 MoDora 支持 multi\-document management and cross\-document analysis，但从代码细节看，这种支持在不同检索分支中的完整程度并不一致。\(GitHub\)

---

## 16. 端到端代码级流程重构

综合以上代码，可以把 MoDora 的端到端流程写成下面这条更细的数据流。

### 16\.1 文档处理阶段

```Plain Text
输入 PDF
  ↓
process_document_task()
  ↓
尝试加载 OCR model
  ├─ 成功：model.predict_iter(pdf)
  └─ 失败：extract_pdf_blocks(pdf)
  ↓
OcrExtractResponse(source, blocks)
  ↓
保存 ocr.json
  ↓
get_components_async()
  ↓
StructureAnalyzer.analyze()
  ├─ title block:
  │    flush 当前 text component
  │    flush non_text_cache
  │    初始化新的 text component
  │
  ├─ normal text block:
  │    追加到 cur_text_co.data
  │    追加 bbox 到 cur_text_co.location
  │
  ├─ image/chart/table block:
  │    创建非文本 Component
  │    根据前后 figure_title / vision_footnote 判断标题
  │    放入 non_text_cache
  │
  ├─ header/footer/number/aside block:
  │    按页聚合到 supplement
  │
  └─ 遍历结束:
       flush 当前 text component
       flush non_text_cache
  ↓
ComponentPack(body, supplement)
  ↓
EnrichmentService.enrich_async()
  ├─ 遍历 co_pack.body
  ├─ 只处理 image/chart/table
  ├─ 根据 location 截图
  ├─ LLM 生成 [T] title / [M] metadata / [C] content
  └─ 回填 Component
  ↓
保存 cp.json
  ↓
build_tree_async()
  ↓
AsyncLevelGenerator.generate_level()
  ├─ 收集 text component titles
  ├─ 裁剪标题区域图像
  ├─ LLM 输出 Markdown 风格层级标题
  └─ 解析 # 数量为 title_level
  ↓
TreeConstructor.construct_tree()
  ├─ text component 按 title_level 用 stack 建树
  ├─ image/chart/table 挂到当前章节节点下
  └─ supplement 作为 Supplement 子树挂到 root 下
  ↓
AsyncMetadataGenerator.get_metadata()
  ├─ DFS 自底向上
  ├─ text node: generate metadata + integrate child metadata
  ├─ root node: integrate child metadata
  └─ other node: metadata 为空则使用 data
  ↓
CCTree
  ↓
保存 tree.json
  ↓
更新 knowledge_base.json
```

---

### 16\.2 查询阶段

```Plain Text
输入 query
  ↓
QAService.qa()
  ↓
extract_location(query)
  ├─ LLM 输出 page_list
  └─ LLM 输出 position_vector = [row, column]
  ↓
判断检索路径
  ↓
如果有位置线索：
  LocationRetriever
    ├─ 解析目标页
    ├─ 遍历 CCTree
    ├─ 找出指定页上的 node.location
    ├─ bbox 归一化到 [0,1]
    ├─ 与 3×3 网格区域做 overlap
    └─ 命中则写入 RetrievalResult

如果没有位置线索：
  SemanticRetriever
    ├─ 从 root 开始递归
    ├─ check_node_mm 判断当前节点是否相关
    ├─ 收集当前节点 data/location
    ├─ LLM 根据直接子节点 metadata 选择 children
    └─ 递归进入多个被选中的子节点

  optional VectorRetriever
    ├─ 收集所有有 data 的节点
    ├─ 构造 path + metadata + data 文本
    ├─ Chroma embedding 检索
    ├─ rerank
    └─ top_k 写入 RetrievalResult

  合并：
    result.update(semantic_result)
    result.update(vector_result)
  ↓
得到 RetrievalResult
  ├─ text_map
  ├─ locations
  ├─ locations_by_path
  └─ locations_by_file_page
  ↓
裁剪命中 bbox 对应图片
  ↓
reason_retrieved(query, schema, evidence, images)
  ↓
answer
  ↓
check_answer(query, answer)
  ├─ 有效答案：返回
  └─ 无效答案：
       单文档 → reason_whole()
       多文档 → 不支持 whole-page fallback
  ↓
_format_retrieved_docs()
  ├─ 按 file_name + page 聚合 bbox
  ├─ 标注 retrievers: semantic/vector/location
  └─ 生成 retrieved_documents
  ↓
update_impact(path)
  ↓
返回：
{
  answer,
  retrieved_documents,
  node_impacts,
  retrieval_trace
}
```

---

## 17. 疑问解答

1. `co_pack.body` 是正文组件列表，不是一个融合后的大组件。flush 到 `co_pack.body` 是把组件提交到列表中，而不是把文本组件和非文本组件合并成同一组件。

2. MoDora 的组件聚合主要发生在 OCRBlock 到 Component 阶段：多个普通文本 block 拼接成一个 text component；同页 header/footer/number/aside block 聚合成 supplement component；image/chart/table 保持为独立非文本 component。

3. Enrichment 发生在 `StructureAnalyzer` 之后，且只处理 `image/chart/table`。它不会改变组件之间的独立性，只是为非文本组件补充 title、metadata、data。

4. `AsyncLevelGenerator` 之前，各 text component 虽然有 title，但 title\_level 基本只是默认值 1；真正的标题层级由 LLM 根据标题列表和标题区域截图生成。

5. 查询中的 page list 和 position vector 不是用户直接输入的，而是 LLM 从自然语言问题中解析出来的。例如“第一页右上角”会被转换成 `Page: [1]; Position: [1,3]`。

6. LocationRetriever 的本质是：把自然语言位置映射到 3×3 网格，把组件 bbox 归一化到 `[0,1]`，再判断 bbox 与目标网格是否重叠。

7. LLM\-guided pruning 每次只让 LLM 看到当前节点的直接子节点标题和 metadata；如果多个子节点相关，会同时进入下一层递归。

8. SemanticRetriever 与 VectorRetriever 是两条独立召回通道，最终通过 `RetrievalResult.update()` 做简单 union，而不是复杂的 rank fusion。

9. `check_answer()` 不是严格事实验证，而是判断回答是否为有效回答；如果回答类似 None/N/A/无法回答，则触发 fallback。

10. `retrieved_documents` 是把命中 bbox 按文件名和页码聚合后的证据列表，方便前端或后续模块在同一页高亮多个证据区域。

11. MoDora 代码支持多文档树合并和多文档 semantic/vector 检索，但 location retrieval 和 whole\-document fallback 当前仍主要是单文档实现。



## 18. 关键结论

1. 摄入数据必须是 pdf ，Modora 全流程均是基于 pdf ，且 Modora 确实对于图表有更多的处理（描述增强、节点挂载、截图取证）

2. CC Tree 的构造基础：由 LLM 结合标题文本和标题区域视觉信息生成的标题层级

3. CC Tree 支持多文档，多文档下的CC Tree和文件系统结构很像，但是是存储的json文件，而不是原生文件系统

4. Modora 的 semantic 检索器基于 CC Tree 进行检索，检索时也是渐进式一层一层地展开，递归处理每一层的节点，也是 agentic 实现（是循环多轮，但不是tool调用，每一轮固定调用`_is_relevant()`）

5. 检索时， ov 可基于向量相似度给出多个目录候选（search），而 Modora 似乎固定从 CC Tree 的根结点开始检索

6. MoDora好像和PageIndex比较像

7. 复现相关：

    - 三种检索器：location 应该与我们想做的实验无关；semantic 强相关；vector 是可选功能，需要考虑是否启用

    - 数据集需要全部处理为 pdf 才能摄入



