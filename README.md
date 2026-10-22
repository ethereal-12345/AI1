Token（元词）
- Token：Tokenizer切出来的数字id，是模型能看懂的最小单位

- Embedding：把token id变成向量；位置编码给向量补上顺序信息

- Q=我要找什么，K=我有什么，V=我携带什么信息；自注意力=用Q‑K相似度加权V

- 多头注意力：多组QKV，捕捉多种关系；注意力=token之间互相通信

- FFN：每个token独立非线性变换，自己消化信息

- 因果mask：不让看到未来token，实现自回归生成

- Encoder‑only：理解；Encoder‑Decoder：翻译式生成；Decoder‑only：现在LLM主流

- O(n²)复杂度 + KV‑Cache加速 + 上下文窗口长度限制，是LLM工程性能的根源

主流模型对比
- Base基座：不会聊天，适合微调；Chat/Instruct：可直接对话

- GPT：闭源标杆，原型神器，不可本地，注意合规

- Llama3：开源生态之王，中文原生一般，商用有条款

- Qwen2：Apache‑2.0友好商用，中文强，国内私有化首选

- Yi：长上下文很强；DeepSeek：代码&逻辑突出；Mistral/Mixtral：速度快，中文弱

- MoE：总参大，每次只激活部分专家；GQA：优化KV‑Cache显存

- SWA滑动窗口：用有限注意力换超长文本

- 选型三要素：许可协议、中文能力、上下文长度、你的硬件/隐私约束

- 大模型Apl调用
- 一、基础概念
 
- Endpoint（端点）：调用地址，就是那一串  https://xxx/v1/chat/completions 

- API‑Key：你的身份凭证，相当于「账号+令牌」，绝对不能硬编码提交到 Git

- Model 参数：指定要用哪个模型（如  gpt-4o-mini 、 qwen2-7b-instruct ）

- Prompt 不再是单个字符串：现代对话接口用 messages 消息数组

- Token 计费：一般按「输入token + 输出token」一起扣费，不是只算回答
 
两个容易搞混接口：
-  /completions ：旧版，纯续写（现在基本废弃）
-  /chat/completions ：新版对话接口，我们现在几乎只用这个

  
- messages的结构
- [
  {"role":"system", "content":"你是一个有用的助手"},
  {"role":"user", "content":"什么是RAG"},
  {"role":"assistant", "content":"RAG即检索增强生成…"}
]

system ：系统提示，设定人设、规则、输出格式（有些开源推理服务对 system 支持弱）

-  user ：用户消息

-  assistant ：模型历史回答
 
注意：顺序很重要，就是对话上下文；全部会被转成 token 塞进上下文窗口
坑：不是所有模型都原生认 system；部分国产/开源模型要求把指令塞到第一条 user，硬写 system 会效果变差。

主要用  /v1/chat/completions ，用 messages 数组承载完整对话上下文

- temperature 控制随机性，不是智商；stream=真 是 SSE 打字机，不加速生成

- openai SDK 不只调用 OpenAI：所有兼容 OpenAI 协议的服务，都可以复用

- Tool‑call：模型只生成调用意图和参数，执行工具必须是你后端代码完成

- key 不能硬编码，用 .env + 环境变量，加入 gitignore

- 429=限流，要指数退避重试；必须设置超时防止卡死

- token 不是字符，不能用 len() 估算；要警惕上下文窗口溢出

- vLLM/Ollama 本地推理也可以输出 OpenAI 兼容接口，实现业务层和模型解耦


OpenAl/DeepSeek结构化输出
- 三种结构化：纯prompt软约定、json_object（保证是JSON，不保证字段）、json_schema（带schema约束）

- OpenAI 的 strict=True 是真强约束，但必须给 schema 一个 name

- DeepSeek 兼容接口，但严格模式力度弱，容易多出 markdown 标记，记得清洗

- 结构化任务温度建议 0‑0.3，不要高温度

- Pydantic 定义模型 → 自动导出 schema，手写 schema 容易出错

- 不管平台说有多“强约束”，都必须再用 model_validate_json 校验

- 增加最多2‑3次有限重试，是生产环境低成本提高成功率的手段

- stream 不适合用来做结构化解析

- OpenAl  vs DeepSeek结构化对照
- - OpenAI gpt‑4o / gpt‑4o‑mini

- ✅ 完整  json_schema + strict: true ，推理层强约束

- ✅ 强制需要 schema 的 name

- ✅ 畸形JSON概率很低，但仍建议二次校验

- ⚠️ strict模式不允许额外字段

- DeepSeek‑chat / deepseek‑v2

- ✅ 支持  json_object （最稳）

- ⚠️ 支持 json_schema，但 strict 约束力度不如 OpenAI

- ⚠️ 偶尔输出 ```json 包裹，需要正则清洗

- ✅ name 不是强制，但写上无害
 
工程策略：
- 如果是原型/对稳定性要求高：用 OpenAI strict json_schema
- 如果要用 DeepSeek：优先  json_object  + 清洗 + Pydantic + 有限重试，性价比最高

- 结构化输出最佳实践/大坑
- ✅ 最佳实践
 
- 做抽取/分类/结构化时，温度压低：0 ~ 0.3，不要用 0.7；随机性越大，JSON越容易崩

- 在 Field(description=...) 里写清楚每个字段含义、取值范围、例子，模型理解会好很多

-  enum  尽量用，能大幅减少非法值

- 永远不要信任“平台说的强约束”，必须再用 Pydantic model_validate_json 校验

- 加上有限次数重试（最多2‑3次，不要死循环）

- 写兜底日志：失败时把原始返回存下来，方便排查为什么解析炸了

- ❌ 高频踩坑
 
1. 以为开个  json_object  就会自动保证字段、类型正确 → 它只保证整体是JSON

2. OpenAI 使用 json_schema 忘记写  name ，直接 400 参数错误

3. 开启 strict=True，但 schema 里允许  additionalProperties: true ，直接报错

4. 用 DeepSeek / 自建 vLLM，期待和 OpenAI strict 一样强的硬约束 → 实际约束弱，还可能带 ```json

5. 温度开很高（>1）还想要稳定JSON → 很容易畸形

6. 只校验 json.loads 能解析，不用 Pydantic 校验字段类型、范围、必填

7. schema 写得太复杂、嵌套过深，模型跟不上，不断生成残缺结构

8. prompt 还在说“可以简单解释一下”，同时又要纯JSON → 指令自相矛盾

9. 把 stream 和结构化输出一起用：流返回的是片段，很难直接当成完整JSON解析，结构化尽量关 stream 
补充：vLLM 也支持  response_format ，但它目前只实现了 json_object，不支持完整 strict‑json_schema，只能靠 prompt + Pydantic 兜底。

Token计算与成本优化
- token ≠ 汉字，只能用对应模型的 tokenizer计算；记得带上 apply_chat_template，不要漏模板标记

- usage 以服务商返回为准；本地只能用来预判、预警、前置截断

- 总窗口 = 输入 + 最大可输出；max_tokens 只是输出上限，不是全部窗口

- KV‑Cache 加速 decode，但超长输入的 prefill 算力开销无法避免

- 浪费 token 的大头：无限变长对话、无脑塞大量RAG片段、prompt废话、Agent死循环调用、大模型做简单杂活

- 上下文优化方案：滑动窗口 / 历史摘要 / 相关性过滤；不要粗暴在句子中间截断

- 分层调度：重推理用强模型；分类、摘要、校验交给廉价/轻量模型；能不用LLM就不用

- 缓存高频稳定答案，限制最大重试次数、限制Agent最大工具轮次

- 埋点记录每一次 token 消耗，设置用量与超长请求告警，做到可观测

- 几个常见的误区
- ❌「用超大上下文模型，就可以不用做RAG分块、不用管长度」
✅ 可以，但算力与token成本很高，长上下文召回衰减，收益有限
 
❌「输出token比输入便宜，可以疯狂让模型写超长内容」
✅ 绝大多数服务商输出更贵，不要滥用长生成
 
❌「tiktoken可以用来精确算所有国产模型token」
✅ 不行，每个模型有专属 tokenizer，tiktoken 只适合 OpenAI 系列
 
❌「只要开了KV‑Cache，长序列就很快很便宜」
✅ KV‑Cache 只加速生成阶段（decode）；超长输入的 prefill（预处理）仍然很重，没法跳过
 
❌「优化就是尽量把 prompt 压缩到最短」
✅ 过度压缩导致指令歧义，模型犯错、触发多次重试，总成本反而更高；是去掉无效token，不是干掉必要信息

❌「成本 = 单价 × 调用次数」
✅ 更准确：成本 = 单价 ×（每轮平均输入token + 平均输出token）× 调用次数；单次请求的长度往往比调用次数影响更大

Streaming流式响应处理
 - stream 是感知更快，不是生成更快；传输大多用 SSE（HTTP单向长连接）≠ WebSocket

- delta 是增量片段，不是完整答案，必须内存拼接才能拿到全文

- 不是每个 chunk 都有 content；finish_reason 标记是否结束

- 流式模式大多无法可靠拿到官方 usage，只能本地 tokenizer 估算

- 安全原则：前端永远不能直接调用 LLM API；后端做流式代理转发

- FastAPI 使用  StreamingResponse + media_type="text/event‑stream" ，yield  data:xxx\n\n 

- tool‑call 流式时 arguments 也是增量碎片，必须等流结束再解析执行

- 用户关闭页面 ≈ 连接断开，如不主动取消上游请求，会白白浪费 token

- 结构化输出不要用流式；流式适合面向人类展示长对话

- 流式的生产级痛点/避坑问题
- ✅ 问题1：连接中途断掉（用户关掉页面、网络波动）
 
- 默认我们的后端还会继续向上游 LLM 读完整个完整答案，白白浪费大量 token 和钱！

- FastAPI 可以监听  is_disconnected() ，检测客户端已经断开 → 立刻取消向上游的请求，停止消耗。

- openai sdk 支持设置  timeout ，也支持用  context.cancel()  取消任务。
 
✅ 问题2：拿不到真实 usage（token消耗）
 
- 流式模式下很多服务商不返回 usage，或者只在最后一个 chunk 才给，还经常丢

- 对策：用对应 tokenizer，对「输入 messages + 拼接得到的 full_text」做本地估算，只用于内部监控预警，不能用来和服务商对账，账单以平台为准。
 
✅ 问题3：不要在流回调里写重业务逻辑、数据库写入
 
- 流是持续高速吐出小块，网络抖动、重试、断连都可能发生

- 正确：等流完整结束，拿到完整 full_text，再统一写库、记日志、埋点 token
 
✅ 问题4：超时设置非常重要！
 
- 流式是长连接，默认超时经常很短或者无限，会出现连接僵死

- openai sdk： client.with_options(timeout=120.0)  设置长超时

- 你的 web 服务器（uvicorn / nginx）也要调大 keep‑alive / proxy 超时，不然中间件会掐断长连接
 
✅ 问题5：不要假设所有服务商的流格式 100% 和 OpenAI 一模一样
 
- 有的流不会发  [DONE] 

- 有的 chunk 没有 choices

- 有的 finish_reason 字段名字不一样
→ 一定要加防御判断： if chunk.choices and chunk.choices[0].delta.content ，不要直接硬索引
 
✅ 问题6：流≠万能，不要什么都流式
 
- 纯抽取、结构化、校验、小回答，流式收益很低，还增加解析复杂度

- 流式最适合：给人看的长文本对话输出，提供打字机体感
 
❌ 常见错误大坑：
- 以为 stream=True 会加速模型计算

- 把 delta 当成完整答案，每一块直接存库

- TCP 粘包时不做 buffer，自己手写简陋 SSE 解析就上生产

- 流式 tool‑call，中途就尝试解析残缺 arguments

- API‑key 暴露给前端，让浏览器直接去调用 LLM 服务商（严重安全漏洞）

- 客户端断开，但后端没有取消上游请求，持续浪费 token

- 忘记设置超时，长连接无限挂住，连接池被占满



- 结构化提示词设计
- - 结构化提示词核心：限制模型自由度、固定流程、固定输出

- 六层架构：角色、任务、输入约束、执行步骤、输出结构、禁止规则

- 三级结构化：纯prompt软约束 → json_object → json_schema强约束

- 结构化任务温度必须 0~0.2

- 必须定义：字段说明、类型、枚举、空值策略

- 必须：模型结构约束 + 代码校验双重兜底

- 禁止多余话术、禁止脑补、禁止自创字段

- 四大万能结构化prompt模板
- 信息抽取模板
【角色】你是专业信息抽取助手，严谨、客观、不编造。
【任务】从用户输入文本中提取指定字段信息。

【执行流程】
1. 仔细阅读原文，只依据原文内容提取，禁止脑补
2. 无对应信息填空字符串或null
3. 严格按照输出结构输出，不添加任何解释

【输出结构】
{
  "name": "姓名，无则空",
  "phone": "手机号，无则空",
  "intent": "用户意图分类：咨询/投诉/建议/其他"
}

【禁止】
禁止解释、禁止markdown、禁止多余文字


分类模板
【角色】精准分类器
【任务】将用户问题归入唯一类别

【分类枚举】
A：咨询问题
B：功能使用
C：投诉建议
D：其他

【规则】
1. 只能输出以上四类，不可自创
2. 模糊内容归为D
3. 只输出JSON，不解释

【输出】{"category": ""}


摘要+关键词模板
【任务】对文本生成摘要与关键词
【规则】
1. 摘要80字以内，保留核心信息
2. 关键词3-5个
3. 不编造原文没有的内容

【输出结构】
{"summary":"", "keywords":[]}


思维推理模板
【执行流程】
1. 先拆解问题条件
2. 分析已知、未知
3. 分步推理
4. 给出最终答案

【输出结构】
{
  "reason": "分步推理过程",
  "answer": "最终结论"
}

七、高频踩坑清单（上线必避）
 
❌ 坑1：只说“输出JSON”，不定义字段规则
→ 字段乱、漏、随机
 
❌ 坑2：不定义空值策略
→ 有的返回""、有的返回null、有的不写字段，代码直接崩
 
❌ 坑3：Prompt里又让解释、又让纯JSON
→ 指令冲突，模型随机选择
 
❌ 坑4：字段没有枚举约束
→ 分类值五花八门：“咨询一下”“我想问问”全部不统一
 
❌ 坑5：温度太高（>0.5）做结构化
→ 随机性爆炸
 
❌ 坑6：依赖模型自觉，不加代码校验
→ 生产必炸
✅ 标准答案：
模型层结构化 + 代码层结构化双重保险


Few-shot Learning(少样本学习）
- Few‑shot 基于 ICL 上下文学习：不改变模型权重，只在上下文给示范

- Zero‑shot：无样例；One‑shot：1个；Few‑shot：2‑5个；≠ 微调

- 模型更倾向模仿样例，而不是遵守文字指令，样例质量 > 样例数量

- CoT‑Few‑shot：给「输入 → 推理过程 → 答案」，提升复杂逻辑正确率，但更耗输出token

- 静态Few‑shot：样例固定写死在prompt；动态Few‑shot：用向量检索拿到和当前query相似的样例

- Few‑shot适合快速迭代；调用巨大、任务完全固定，可以考虑微调降低token成本

- 黄金原则：样例尽量来自真实业务、格式100%统一、多样性足够、不要堆砌

- 注意：ICL只能引导模型发挥已有能力，不能凭空教会新知识

- Few-shot设计的8条黄金工程原则
- 1. 样例必须和真实线上数据分布一致
不要用你自己编的“理想句子”，尽量拿真实线上样本；编造的样例经常和真实分布脱节，上线直接拉胯。

2. 样例的格式、标签、写法必须 100% 统一
如果一个样例输出带  ```json ，另一个不带，有的标签写全称有的写缩写，模型就会混乱。
记住：样例示范 > 文字指令

3. 不要堆砌样例，2‑5 个通常足够
除非经过实验证明更多样例确实提升指标，否则不要无脑塞 10+，token 成本爆炸。

4. 样例要具备多样性，不要几乎一模一样
尽量覆盖：正常、边界、模糊、简短、啰嗦等典型场景；避免全部是同一种句式。

5. 样例里不要放模型不该看到的“备注、注释”
模型会误以为注释也是输出的一部分，学着给你加注释。

6. 如果用 CoT‑Few‑shot：样例的推理粒度要统一
不要有的样例推理写 2 句话，有的写 20 句，不然输出长度不可控。

7. 静态样例适合简单任务；复杂多变任务考虑「检索式动态 Few‑shot」
但要给检索样例数量设置硬上限，并且对样例库做质量巡检。

8. 样例本身也要校验
如果样例标签/结构有错，就是「坏示范」，会带来系统性错误。

常见的大坑
❌ 坑1：样例太多，prompt超长，token成本暴涨 + prefill很慢
❌ 坑2：指令写得很完美，但样例本身违反指令 → 模型模仿坏样例
❌ 坑3：样例是自己凭空编的，和真实业务输入完全不是一个风格，线下测试很好，线上翻车
❌ 坑4：动态Few‑shot检索到不相关甚至错误的样例，把模型带偏（叫「样例误导」）
❌ 坑5：误以为 Few‑shot 可以教会模型它本身就不会的知识
 
ICL 是引导它使用已有的能力，不是教它新知识；它不会的，给再多样例也很难学会
❌ 坑6：CoT‑Few‑shot 不加限制，推理无限变长，completion token 失控
❌ 坑7：把 Few‑shot 和结构化输出、json_schema 对立
正确做法：结构化提示词 + 高质量 Few‑shot + json_schema + Pydantic校验，四层防护



Chain-of-Thought
什么是COT
Chain‑of‑Thought（思维链）：不让模型直接给出答案，强制它把「中间推理步骤」写出来，再得出最终结果。
 
大模型不是真的“会思考”，它是根据前面的 token 往下续写；如果直接要答案，它容易跳步、凭直觉猜（也就是「幻觉 / 算术错 / 逻辑漏项」）。
CoT 就是给它一条“推理路径”，让它一步步生成，再从路径走到答案。
 
注意：CoT 不改变模型权重，属于提示工程层面的技术，是「推理引导」，不是微调。

Cot什么时候有用/什么时候没用
✅ 适合：
 
- 多步数学计算、逻辑推理、因果分析

- 需要拆解复杂问题、方案论证、排查原因

- 模型经常“直接蹦一个错答案”的场景
 
❌ 不适合（加了反而不好）：
 
- 简单事实问答、翻译、摘要、简单分类

- 追求极快响应、严格控制 token 成本

- 模型本身能力太弱，只会“形式上假装一步步说，但逻辑还是崩”
 
经验：不是所有任务都要 CoT；能一次答对就不要强行加推理链，白白花钱和耗时

cot应用时遇到的坑
1. 推理对，答案抄错：步骤全正确，但最后数字/结论写错（“正确路径，输出漂移”）

- 对策：写完推理后，再加一句：「请根据上面全部推理，重新验算一次，再输出最终结果」

2. CoT 输出格式不可控：有的时候它不写“思考：”，直接给答案

- 对策：配合 Few‑Shot + Pydantic / JSON 结构化输出，强制字段： reasoning: str, answer: any 

3. token 暴涨：输入+输出都变长，QPS 高的时候成本压力很大

- 对策：线上可做双模式：简单问题直接回答；复杂问题才启用 CoT

4. 幻觉不会被 CoT 彻底消灭
CoT 可以减少幻觉，但不能根治；它可以“一本正经地一步步幻觉”

5. 不要迷信  let's think step by step ：不同模型对这句触发效果不一样，DeepSeek / 国产大模型，用中文指令往往比英文咒语更稳


- CoT：不要直接答，先写推理步骤，再给答案

- Zero‑Shot‑CoT：不加样例，只加「一步步思考」

- Few‑Shot‑CoT：给几条「问题→推理→答案」样例让它模仿
- Self‑Consistency：多条链，投票选答案；ToT：多分支树搜索；ReAct：推理 + 调用工具

- 收益：提升复杂逻辑正确率；代价：更多 token、更慢、更贵

- 边界：简单任务没必要上，小模型可能只会“假推理”



ReAct框架
- ReAct = Reasoning（推理） + Acting（行动），不是前端 React！

- 循环：Thought → Action → Observation → 循环 → Final Answer

- Thought：模型想；Action：模型提议调用工具；Observation：程序给的真实结果

- 和 CoT 区别：CoT 只能空想；ReAct 可以走到外部世界拿事实

- 和 Function‑call 区别：function‑call 是调用工具的能力；ReAct 是带观察反馈的循环范式

- 风险：格式崩坏、编造观察、死循环、token 暴涨；要限制最大迭代轮数

- Rest vs Function-Calling 区分
- - Function‑Calling（函数调用）：是大模型提供的原生能力
模型可以输出结构化 JSON，明确  name 、 arguments ，专门用来告诉我们要调用哪个工具；它只输出调用，不带显式 Thought（当然你可以让它加上）

- ReAct：是一种范式/算法思路：「思考‑行动‑观察‑循环」，它可以用 function‑call 实现，也可以用普通文本解析实现
 
关系：
 
- 早期 ReAct：没有 function‑call，靠解析文本字符串提取 Action，很容易格式崩

- 现在工程上：ReAct 思想 + function‑call 结构化输出 + Pydantic 校验，稳定性大幅提升
 
简单一句话：
Function‑call 是“怎么告诉系统要调用工具”；
ReAct 是“什么时候调用、调用完之后怎么继续思考”的完整循环。


高级策略（Self-Consistency/Reflexion)
- Self‑Consistency 自一致性：调高温度生成 N 条独立 CoT，程序投票取众数；准确率↑，但成本×N；适合离线 / 高价值查询，高并发线上慎用

- Reflexion 反思：Agent 执行完后，回看轨迹、写出失败原因与教训，作为记忆，再重试；不是再算一遍，是“复盘经验”；风险：反思本身也可能幻觉

- CoT：单路径；Self‑Consistency：多路径并行投票；ToT：多分支树搜索可剪枝；ReAct：边思考边调用工具；Reflexion：犯错后复盘反思

- 全部都是提示/运行时策略，不需要微调模型权重

- 收益有代价：更强推理 ⇔ 更长时间 + 更多 token + 更高幻觉风险（策略本身也会错）

这几个策略什么时候用
✅ 适合启用增强推理：
- 高价值、容错低：数学、方案论证、复杂多步骤决策

- 普通 CoT / ReAct 经常翻车

- QPS 不高，可以接受更高耗时、更高成本
 
❌ 不适合：
- 简单问答、分类、翻译、普通摘要

- 高并发线上主链路：Self‑Consistency / 完整 Reflexion 直接上会把账单和延迟打爆

- 弱模型：本身逻辑能力差，再多策略也只是“花式瞎编”
 
工程务实思路：分层策略
默认：普通 ReAct
如果检测到这是复杂问题 / 上一次执行可疑失败：才触发轻量反思 / 2‑3 条 Self‑Consistency；而不是全部请求都拉满。

RAG检索增强生产完整流程
- RAG = 检索增强生成，不改模型权重；分为「离线建索引」和「在线问答」两大阶段

- 离线链路：加载 → 清洗 → 切块 → 向量化 → 存入向量库

- chunk 大小 + overlap、文档质量，往往比向量库更重要

- 在线链路：用户问题 → query改写 → 向量粗召回（可搭配BM25混合检索）→ rerank重排序精筛 → 组装增强上下文 → LLM生成 + 溯源

- embedding：把文本变成向量，算语义近似；reranker：query与chunk深度交互，做精准相关性打分

- 常见陷阱：lost‑in‑the‑middle、上下文越狱、脏文档、不同embedding混用、把向量库当原始数据源

- RAG 不是万能：它只能降低幻觉，不能彻底消灭幻觉

- RAG 的简单版本 vs 生产增强版本
✅ 最简 Demo 版
文档 → load → recursive split → bge embedding → chroma → query→向量检索 topk → 拼 prompt → 生成
👉 能跑，但上线很容易翻车。
 
✅ 务实生产版
文档 → load → clean/OCR → smart chunk（带 metadata）→ embedding → milvus/qdrant/pgvector
查询：query → query改写 → BM25 + 向量 混合召回 → rerank重排序 → 上下文压缩 → 防幻觉系统提示 → LLM生成 → 溯源 + 质量校验

RAG 的常见失败
- 召回失败：真正能回答的 chunk 根本没被检索出来（最常见）
原因：切块不合理、embedding 不匹配、query太短、文档质量差

- 召回到了，但排序差：有用的排在很后面，模型看不到（lost‑in‑the‑middle）

- 召回到正确文档，但模型无视文档，还是自己编（上下文越狱）

- 检索到「语义像但是事实无关」的 chunk，模型照着错误证据输出（垃圾进 → 垃圾出）

- chunk 本身有错、脏数据，RAG 只是“忠实地复述错误”
 
记住一句：RAG 不能创造真相，只能复述你库里已经有的东西；库里有错，它就会有理有据地错。




Embedding向量化技术
- Embedding：文本 → 固定长度浮点数向量，语义近，距离近

- 常用度量：余弦相似度，向量最好归一化

- 黄金规则：文档和 query 必须同一个模型

- 不是维度越高就越好，要看训练数据是否适配领域

- 只负责「表征语义相似」，不能判断事实正确性

- 向量库 ≠ embedding：向量库只是用来存 + 快速搜向量

- 稠密抓语义，稀疏抓关键词，混合检索通常更强

调用/本地使用极简示例
1.OpenAl-style APl
from openai import OpenAI
client = OpenAI()

resp = client.embeddings.create(
    input=["今天天气怎么样"],
    model="text-embedding-3-small"
)
vec = resp.data[0].embedding  # 就是向量 list[float]


2.本地Sentence-Transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vec = model.encode("一段要向量化的文本", normalize_embeddings=True)
# normalize_embeddings=True：自动归一化，方便算余弦

Embebdding的典型坑
1. 查询和文档用了不同 embedding 模型 → 向量空间不匹配，检索几乎无效

2. chunk 切得太长，超过模型最大长度，被悄悄截断，语义残缺

3. chunk 切得太短：语义破碎，向量“含义模糊”

4. 忘记归一化，直接算余弦 / 点积，相似度不准

5. 用通用 embedding 去做高度专业领域（比如专利、病历），召回差

6. 认为「相似度高 = 内容正确」：embedding 只能衡量语义相似，不能判断事实对错

7. 向量库索引没建好，数据量大之后检索极慢

8. 重复文档没有去重，生成大量重复向量，浪费存储还干扰召回

9. 把问题直接当 chunk 入库：很多 embedding 在“问句”和“陈述句”上表征有偏移
 
经典现象：召回一堆字面很像、但和用户真正意图无关的内容 → 这就是 embedding 漂移 / chunk 策略不对。


进阶：稠密 vs 稀疏向量
- Dense 稠密向量：就是我们上面说的 embedding，全部维度都有浮点数，抓「语义相似」，但可能漏掉关键词

- Sparse 稀疏向量：大部分维度是 0，只有少数位置有值，类似 BM25 的升级版（比如 BGE‑M3 的 sparse），擅长关键词、专有名词

- Hybrid 混合检索：稠密召回 + 稀疏召回，再合并打分，是现在高级 RAG 的标配
 
再往后：ColBERT 多向量 embedding，一个 chunk 生成多个 token‑level 向量，能做到更细粒度匹配，但存储和查询成本更高。


向量数据库（ChromaDB/Milvus/Qdrant)
基础概念：
- Collection / 集合：相当于 MySQL 的「表」，存放一批同维度、同 embedding 空间的向量
✅ 同一个 collection 里向量维度必须一致，最好也用同一个 embedding 模型生成

- Vector 向量： list[float] ，一般 384 / 768 / 1024 / 1536 维

- ID：每条向量唯一标识，可以自己指定或自动生成

- Payload / 元数据 metadata：标签、来源文件名、页码、时间、文档类型，用来过滤（比如只搜来自某PDF的片段）

- 距离度量 metric：

-  cosine  余弦：RAG 最常用，不受向量长度影响

-  ip  内积（点积）：向量归一化后，等价于余弦，速度更快

-  L2  欧氏距离：值越小越接近，对向量模长敏感，RAG较少首选

- ANN 近似最近邻
高维下暴力全量计算太慢 → 建立特殊索引，牺牲极少量精度，换几十~几百倍速度；
不是“一定找出全局最像”，是“找出大概率很像的一批”

- 索引类型（三大库共用核心索引）

-  FLAT ：暴力全搜，100%精确，数据少可用，千万级完全不行

-  IVF‑FLAT / IVF‑SQ ：聚类分桶，速度提升，需要训练

-  HNSW ：图索引，综合性能最好，工业最常用；内存消耗偏高

-  SCANN ：谷歌改进版，大库高性能；部分库才支持
 
误区：索引建好后再新增数据，有些索引不会自动重训练，要看具体引擎实现。
 
- 过滤 Filter：不是向量过滤，是先按 metadata 筛选，再在子集里做向量检索；适合权限、文档来源、时间范围等

- 分片 & 副本：海量数据时，分片存不同机器；副本用于高可用（只有 Milvus/Qdrant 这类分布式支持）




三个库分别是什么、定位、优缺点
选型口诀：原型用 Chroma；中小体量 Qdrant；大数据+复杂业务 Milvus
 
🔹 ChromaDB
 
定位：轻量、零配置、原型友好，专为 RAG 快速验证设计
 
- 模式：支持内存模式 / 本地文件持久化；原生不是分布式，不适合超大库、高并发

- 优点：

- 安装极简单： pip install chromadb ，不用额外起服务

- API 极简，自带简易 embedding 包装，几行就能跑 RAG demo

- 自带去重、简单元数据过滤

- 缺点：

- 千万级向量性能明显下滑

- 没有完善集群、负载均衡、备份工具

- 生产高并发不推荐，适合学习、POC、个人项目

- 注意：不要直接把 Chroma 当成线上生产数据库；磁盘版在大量写入时偶有卡顿
 
示例极简：
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_or_create_collection(name="docs")
col.add(
    documents=["xxx"],
    metadatas=[{"source":"book.pdf"}],
    ids=["doc‑001"]
)
res = col.query(query_texts=["用户问题"], n_results=3)



Qdrant
 
定位：平衡易用 + 性能，中小‑中型生产首选，RAG社区热度很高
 
- 实现：Rust 编写，性能好；可以单机部署，也支持集群

- 部署方式：docker 一键启动；也有内存模式（不持久化，仅测试）

- 优点：

- HNSW 实现成熟，检索速度快

- 强大 metadata 过滤、条件查询、 payload 索引

- 支持稀疏向量、混合检索（dense + sparse）

- 有 Web 可视化面板；文档友好

- 单机就可以扛百万‑几百万向量，足够绝大多数AI应用

- 缺点：

- 超大规模（数亿+）集群运维复杂度上升，生态不如 Milvus 成熟

- Rust 底层，出深度问题资料相对少一点
 
适合：创业项目、正式业务、百万级向量、不想太重运维
 
🔹 Milvus（Zilliz）
 
定位：企业级分布式向量数据库，面向大数据量
 
- 基于 etcd + pulsar + minio / s3 做元数据、日志、对象存储；可以水平扩容

- 优点：

- 支持亿级~十亿级向量，可分片、可弹性扩缩容

- 索引种类最全，有完整监控、备份、权限、多租户

- 国内社区活跃，中文文档完善；很多大厂生产在用

- 缺点：

- 重，组件多，单机也需要 docker‑compose 启动一堆服务

- 资源消耗大；小数据集上反而显得“杀鸡用牛刀”

- 配置参数非常多，调优门槛高
 
适合：海量知识库、多租户SaaS、企业级系统
 
补充：还有 PGVector —— 不是独立向量库，是 PostgreSQL 的插件；如果你已经在用PG，向量不多时可以直接用，省去维护新数据库，但超大数据性能不如上面三个。



向量库完整工作链路（RAG视角）
 
离线建库
 
1. 原始文档 → 解析 → 切 chunk → 清洗

2. 调用 embedding 模型生成向量

3. 带上 id、chunk文本、metadata，写入向量库

4. 向量库根据配置，后台建立 ANN 索引
 
在线查询
 
1. 用户 query → 同样 embedding（⚠️必须同一个模型）

2. 向量库：可选先按 metadata filter 过滤子集

3. 在子集上执行 ANN 检索，返回 top‑k 候选

4. （可选）送给 Reranker 重排序，再筛一次

5. 把最终片段组装成上下文，丢给 LLM 生成答案
 
👉 向量库不负责 chunk 切割、不负责 embedding、不负责重排，很多初学者会把这些功能当成向量库自带。


向量数据库常见大坑
1. 同一个 collection 混入不同 embedding 模型产出的向量
→ 向量空间不一样，检索彻底失效；尽量一个 collection 对应一个模型

2. 以为存了向量，就存了全部数据：部分库允许只存向量+id，原始文本需要自己另外维护，查询拿不到原文

3. HNSW 参数乱调： ef_construction （建图时）、 ef_search （查询时）

- ef_construction 越大，索引质量越好、但建索引越慢、占内存越高

- ef_search 越大，检索召回率越高、速度越慢；不能小于要返回的 top‑k

4. 把「metadata过滤」当成万能：过滤是先筛再搜，不是搜完再筛；如果筛完集合很小，召回会变差

5. 向量库持久化 ≠ 自动备份：一定要手动做备份，不要只靠磁盘文件

6. 数据量大了忘记重建/刷新索引，检索越来越慢

7. 迷信“向量库越强，RAG效果越好”：
召回质量80%取决于 chunk策略 + embedding，向量库主要保证「检索快不快」，不决定“能不能搜到对的”

8. ANN 近似检索坑：top‑k 有可能漏掉真正相关文档；可以多召回几条（比如先取 top‑15，再 rerank 留下 top‑3）

9. 直接存超大文本到 payload：向量库 payload 不适合当全文搜索引擎，长文本尽量只存摘要/引用路径，原文放对象存



生产级最佳实践
- 开发/原型：优先 Chroma 快速验证逻辑；不要直接上 Milvus

- 准备上线：优先 Qdrant 单机；数据持续涨到几百万再评估集群

- 多租户知识库：尽量按业务/租户划分 collection 或者带上 tenant_id metadata 过滤

- 写入：批量插入，不要循环单条 add；单条写入性能很差

- 定期：去重、清理失效文档、重建索引、做备份

- 检索策略：不要只靠纯稠密向量；有条件做 稠密 + BM25 / 稀疏向量 → 混合召回，再 reranker 重排

- 监控：关注召回率、查询延迟，而不是只看“页面能返回结果”


混合检索
为什么需要混合检索
先回顾两种单路召回的短板：
- 纯语义检索（Dense / Embedding）
✅ 擅长同义、改写、意译；不懂字面也能懂意图
❌ 短板：对专有名词、型号、编号、专业缩写、精确数字、生造词容易“视而不见”；语义漂移，召回一堆感觉很像但关键词不对的内容

- 纯关键词检索（BM25 / 稀疏向量）
✅ 擅长字面命中，专有名词、精确词很强；不依赖向量训练
❌ 短板：不会理解同义改写；用户换个说法就搜不到；容易被无关字面词干扰
 
👉 混合检索 = 两路（或多路）独立召回 → 合并 → 重新打分 → 再选出最终候选
目标：同时拿到语义相关性 + 关键词精确性，互相补漏
 
⚠️ 重点误区：
不是「先关键词搜一遍，再在结果里向量重排」，也不是「向量搜完再关键词过滤」——那不是标准 hybrid。
标准：两路各自在全库上召回，再融合结果。

关键词检索：BM25到底是什么
BM25 是现在工业最主流的关键词打分算法，是 TF‑IDF 的升级版。
 
- TF：词在当前文档出现频次

- IDF：这个词在整个库里有多稀有（越稀有权重越高）

- k1、b：调节参数；k1控制词频饱和速度，b控制文档长度惩罚
 
打分逻辑：词命中越多、词越罕见、文档不长 → 分数越高
 
注意：BM25 不是“有没有这个词”，是一个连续相关性分数，不是布尔命中。
 
局限：
 
- 只看词形，不认识同义词

- 对分词非常敏感：中文必须做好分词；分词烂，BM25直接废掉
 
实现形态有两大类：
 
- 传统 BM25：Elasticsearch / Jieba + rank_bm25（本地轻量）

- 学习型稀疏向量：BGE‑M3‑sparse、SPLADE，本质是「神经网络算出来的稀疏权重」，可以理解为AI增强版BM25
 
rank_bm25：适合Demo；生产中文不要裸用，必须先做中文分词+停用词过滤。



混合检索完整标准流程
用户Query
    ↓
┌─────┴─────┐
↓           ↓
Query→Dense ｜ Query→BM25 / Sparse
↓           ↓
全库稠密召回Top‑K1   ｜ 全库关键词/稀疏召回Top‑K2
    ↘️      ↙️
去重（同一个chunk不要出现两次）
    ↓
分数融合 / 重排序
    ↓
选出最终 Top‑N 给到 LLM
实操经验：K1、K2可以比最终需要的多一点，例如各召回 12‑20 条，合并去重后大概20‑30个候选，再处理。
 
关键点：两路分数值域不一样，不能直接相加！
 
- BM25分数：可能 0 ~ 几十，范围不稳定，和库大小有关

- 余弦相似度：[-1, +1]，大多落在 [0,1]
直接  score_dense + score_bm25  会被值域大的那一路完全“碾压”。
 
👉 所以必须先归一化，再加权。



三种主流分数融合策略
 
① Reciprocal Rank Fusion，RRF 倒数秩融合【最简单、最稳，首选】
 
不直接用原始分数，只用“排在第几名”这个位置，规避值域不一致的大坑。
 
公式简化理解：
 score = Σ 1 / (rank + k) 
k 是平滑常数，一般取 60（很多库默认就是60）
 
- rank=1（第一名）：得分很高

- rank越大，分数快速衰减

- 两路各自排好，再按RRF重新算总分，再整体重排
 
✅ 优点：不需要调权重、不需要归一化、鲁棒极强；大部分项目默认用 RRF
❌ 缺点：完全丢掉原始相似度的强弱信息，只信任顺序
 
Qdrant、Milvus、LangChain 都已经内置支持 RRF。
 
② Weighted Sum 加权和（需要先 Min‑Max 归一化）
 
先把两路分数都缩到 [0‑1]：
 final_score = α * norm_dense + β * norm_bm25 
 
- α、β 是权重，比如 α=0.7，β=0.3（语义略重，关键词兜底）

- 如果是技术文档、有大量编号/专有名词，可以调到 α=0.5，β=0.5
 
⚠️ 坑：权重不是万能，不同数据集最优权重不一样，没有通用黄金值，最好有评测集才能调好。
如果没有评测集，不建议上来就硬加权，优先 RRF。
 
③ 交给 Reranker 重排序融合（最强，但最贵）
 
两路召回得到候选池（比如25‑30条），不管原来分数，全部丢给交叉编码器 Reranker，让它重新独立打分排序。
 
链路：
 Dense top20 + BM25 top20 → 去重得到 ~30个候选 → Reranker 重排 → 取 top3‑5 
 
✅ 效果通常最好
❌ 代价：reranker 推理有耗时、有算力成本；候选不能放太大，不然很慢
 
工业最佳组合：Hybrid(RRF) + Reranker 二次提纯


六、常见实现方案（你写RAG时会碰到）
 
方案A：Elasticsearch（BM25） + Milvus/Qdrant（Dense）
 
- ES负责 BM25 召回；向量库负责 dense召回；业务层拿到两份结果，自己做 RRF 融合

- 优点：成熟，ES分词+BM25很强；适合存量ES系统

- 缺点：两套存储要维护，文档要双写，同步是噩梦；新增、修改、删除要两边同时更新，容易数据不一致
 
方案B：单向量库内完成（BGE‑M3 dense + sparse）
Milvus / Qdrant 支持一条文档同时存入稠密向量 + 稀疏向量；查询同时 query dense + sparse，库内部做 RRF 融合。
 
- 优点：只存一份数据，不用双写ES，维护简单；现在新项目主流

- 缺点：稀疏向量的质量依赖 embedding模型，中文分词不如专业ES精细；需要支持 sparse 的 embedding（BGE‑M3、gte‑m3）
 
方案C：轻量 Demo：rank_bm25 + Chroma
 
- rank_bm25 在内存里跑 BM25，Chroma 做 dense；代码层自己实现RRF

- 优点：不用额外部署服务；适合学习、POC

- 缺点：数据量大后内存爆炸，不能上生产
 
⚠️ Chroma 原生不支持稀疏向量与内置混合检索，只能业务代码手动实现两路召回再融合。



RRF 极简可运行示例
def rrf(ranks, k=60):
    from collections import defaultdict
    scores = defaultdict(float)
    for rank_list in ranks:
        for idx, doc_id in enumerate(rank_list):
            # rank从1开始
            rank = idx + 1
            scores[doc_id] += 1.0 / (rank + k)
    # 按总分降序返回
    return sorted(scores.keys(), key=lambda x: -scores[x])

# dense召回得到id列表
dense_ids = ["c01","c03","c07","c02"]
# bm25召回得到id列表
bm25_ids = ["c03","c09","c01","c05"]

final_ids = rrf([dense_ids, bm25_ids])


八、常见的陷阱大坑
1. 误以为「混合检索 = 向量检索之后再用关键词过滤」
→ 这不是召回融合，只是过滤；关键词相关的好内容根本没机会进入候选池

2. 直接把 BM25原始分数 和 余弦相似度 相加
→ 值域完全不一样，一路直接把另一路“淹没”，混合等于白做

3. 双写 ES + 向量库，但没有保证数据强同步
→ 更新文档只更一边，两边数据不一致，召回错乱

4. BM25 中文没有做好分词、没有去停用词
→ 切出来一堆单字，BM25完全失效；中文不要直接按字符分割

5. 盲目给 β（关键词权重）拉很高
→ 变成回到纯关键词检索，同义改写全部搜不到

6. 认为「混合检索 = 一定比纯向量强」
如果你的文档没有大量专有名词、编号、代码；混合提升可能很小，甚至轻微下降

7. 混合召回之后就直接丢给LLM，忘记去重
同一个chunk两路都命中，出现两次，浪费上下文token，还容易造成重复回答

8. 稀疏向量不等于万能BM25：BGE‑M3‑sparse对非常冷门自定义专有名词，有时还是干不过ES的BM25
 
一个经验信号：什么时候你该上混合检索？
测试时经常发现：用户带型号/编号/专业术语的问题，明明库里有，纯向量却召回不到。

什么时候用什么策略
 
- 快速Demo / 小知识库：先用纯Dense；发现专有名词召回差 → 手动实现 BM25 + RRF 验证

- 新项目：优先 Qdrant / Milvus + BGE‑M3，dense + sparse，库内混合 + RRF

- 已有 Elasticsearch：可以 ES‑BM25 + 向量库Dense，业务层RRF；注意做好双写同步

- 追求最高召回质量：混合召回得到候选池 → 再交给 Reranker 重排序
 
但记住：混合检索救不了糟糕的 chunk 策略和烂 embedding；如果 chunk 切得完全破碎，混合也回天乏术。



重排序模型（Reranker)
Bi‑Encoder vs Cross‑Encoder
 
这是整个 RAG 检索体系最底层的区分
 
- Bi‑Encoder（双编码器）——就是普通 Embedding 模型

- query 和 chunk 分开独立过模型，各产出一个向量

- 检索时：只算向量距离

- ✅ 速度极快，可以提前把文档向量离线算好入库

- ❌ 没有真正“看到”query和chunk之间的交互；匹配能力上限有限

- Cross‑Encoder（交叉编码器）——就是 Reranker

- 输入格式： [CLS] query [SEP] chunk [SEP] 

- query 和 chunk 在模型内部互相做注意力交互

- 不输出长向量，最后只输出一个 0‑1 的相关性分数

- ✅ 匹配精度显著更高，能识别微妙的“似是而非”

- ❌ 不能提前预计算：query一变，每一条候选都要重新跑一次推理；慢、耗显存
 
形象记忆：
 
- Bi‑Encoder：两个人各自看完自己的纸条，然后远距离猜像不像

- Cross‑Encoder：两个人坐到一起，逐字对照，仔细判断到底有没有关系


Reranker的完整工程流程
用户 Query
   ↓
粗排（Dense / Hybrid混合检索）→ 召回 Top‑15 ~ Top‑30 候选（多召回一点，留余量）
   ↓
全部候选 + query 送入 Reranker，逐条算出新的相关分
   ↓
按 reranker 分数重新降序排序
   ↓
设置一个分数阈值，过滤掉“明显不相关”的chunk
   ↓
取最终 Top‑3 ~ Top‑5，组装上下文送给 LLM
⚠️ 注意：粗排原来的分数直接丢掉，以 reranker 的新分数为准
不是“在原来分数上加一点权重”，是重新判决。
 
为什么粗排要多召回？
Reranker 有机会把粗排里排名靠后、但真正有用的 chunk“捞上来”；但也不能太多：超过40‑50条，速度会明显变慢。

主流 Reranker 模型
大多基于 BERT / RoBERTa 架构，专门用「(query, passage) 是否相关」的数据集做微调： 
-  bge‑reranker‑base‑zh / large‑zh ｜BAAI
✅ 国内最常用，中文效果好，文档完善；768维底座，推理压力适中

-  m3e‑reranker‑base 
✅ 和 M3E embedding 生态配套，适合整套国产链路

-  gte‑reranker‑base‑zh 
✅ 质量不错，社区热度上升

-  cross‑encoder/ms‑marco‑base ｜英文标杆，中文一般，不建议直接用在中文业务

- 闭源：如 CoHere Rerank、DeepSeek Rerank API
✅ 不用部署，开箱即用；❌ 有调用成本、有延迟、数据要出网
 
避坑：不要随便拿一个普通 BERT 当 reranker；必须是已经在相关性数据集上微调过的 Cross‑Encoder，否则输出的分数没有意义。
 
还有一类：LLM‑as‑Reranker
用小LLM让它输出： 这个文档和问题是否相关？分数 0‑5 
✅ 可以理解更复杂逻辑；❌ 速度慢、不稳定、成本高，一般只做实验，极少放标准流水线。



五、部署 & 调用的两种方式
方式1.本地 Sentence‑Transformers 简单跑（适合测试）
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base-zh")

query = "怎么修改Milvus的分片数？"
candidates = [
    "Milvus可以通过alter collection调整分片……",
    "MySQL分片需要修改my.cnf……",
    "向量库HNSW索引参数M代表邻居数量……"
]

pairs = [(query, text) for text in candidates]
scores = model.predict(pairs)

# 把文本和分数打包，按分数降序
res = sorted(zip(candidates, scores), key=lambda x: -x[1])
⚠️ 这只适合开发验证；生产不建议直接在业务进程加载模型，会阻塞、内存爆炸。
 
方式2：生产：独立成 HTTP 服务 
常用方案：
 
-  text‑embeddings‑inference 

-  bge‑reranker 官方 fastapi 服务 

-  vLLM / infinity  同时 serving embedding + reranker
 
业务服务通过 HTTP 去请求这个独立推理服务，实现解耦、可以弹性扩显存实例。
 
Qdrant / Milvus 本身不带 reranker；它们只负责召回；reranker 是我们业务层 / 外部服务完成。
 
补充：部分框架如 LangChain / LlamaIndex 已经封装好 Reranker 组件，但底层逻辑完全一样。


阈值是什么？怎么设？
 
Reranker 返回的分数不是绝对真值，是相对偏好；但可以设一个最低相关阈值，用来“干掉垃圾”。
 
参考经验值（以 bge‑reranker‑base‑zh 为例，仅供起点，需要自己调）：
 
- 0.6‑0.7：大概率高度相关，保留

- 0.3‑0.6：可能部分相关，视情况

- <0.2‑0.3：基本不相关，可以直接丢掉
 
⚠️ 重要：不要硬抄网上阈值！不同模型分数分布不一样，甚至同一个模型，不同业务域分布也会偏移。
正确做法：拿一批真实 query，观察 reranker 在「真相关 / 真不相关」样本上的分数，再定自己的阈值。
 
一个强大操作：允许返回的候选数量是可变的，不是永远固定返回3条；低于阈值就直接扔掉，上下文不够再适当放宽。
 
 
 
Reranker 不能干什么
 
- ❌ 不能“读懂全文，补全缺失信息”：库里没有的知识，它变不出来

- ❌ 不能修复切得稀烂、语义破碎的 bad chunk

- ❌ 不能修正 embedding 完全失效（比如用错模型）

- ❌ 不是召回器：不能去遍历全部向量库

- ❌ 不能保证事实正确：只能判断「这一段和问题像不像」，不能判断这段话是不是事实
 
✅ 它真正能干的：
在粗排捞出来的候选池里，把“看着像但其实无关”的噪声压低，把藏在后面的真相关片段往上提。
 
比喻：粗排是海选，挑30个候选人；Reranker 是面试官，仔细面试一遍，选出最靠谱3个人上场。
但如果海选一个对的都没捞到，面试官再厉害，也选不出正确答案。
 
👉 Reranker 救不了召回完全失败，只能改善召回质量。
 
 
 
完整黄金 RAG 流水线（现在可以串起来了）
 
这是目前工业上非常主流的成熟链路：
 
Query
→ Query改写（可选：扩写 / 分解子问题）
→ 混合召回 Dense + Sparse/BM25 + RRF融合，取 top20‑30
→ Reranker 精排 + 分数阈值过滤
→ 去重、长度裁剪，控制总上下文token
→ 注入系统提示词，送给LLM生成
→（可选）做答案校验
 
⚠️ 但注意：不是所有小项目都必须堆满全套
 
- Demo / 小知识库：纯 dense + top5 就能跑

- 当你发现：经常“搜到了但是排在很后面、被漏掉”，这时再加 reranker 收益最大

- 资源有限时：先优化 chunk + embedding；再上混合检索；最后再补 reranker；不要反过来
 
 
 
高频工程坑
 
1. 拿 Reranker 去全库检索
几十万个文档全部跑一遍 reranker，延迟几十秒，显存直接打爆，完全不可行

2. 粗排召回太少（比如只召回 top3），再给 reranker
池子里面本来就没正确答案，reranker 再强也捞不出来

3. 粗排召回太多（>50）丢给 reranker
推理时间暴增，线上超时，甚至OOM

4. 直接复用 Bi‑Encoder 的相似度当精排依据
等于白做精排；要用 reranker 新产出的分数

5. 照搬别人的阈值，不看自己数据集分数分布
要么丢大量有用内容，要么留下一堆垃圾噪声

6. 把 Reranker 和 LLM 塞在同一个业务进程加载
进程卡死、显存冲突，生产大忌，推理服务要独立

7. 期待 reranker 解决幻觉：它只能挑相关片段，不能保证片段本身是事实正确

8. 测试时只看“能不能答出来”，不看 reranker 分数：上线后遇到长尾 query 直接翻车
 
 
 
Reranker 性能开销参考
 
- bge‑reranker‑base‑zh：单条推理几‑十几毫秒（取决于硬件+长度）

- 如果一次25个候选：一次 rerank 大概 200‑400ms

- large 版本质量更高，但速度差不多慢一倍
 
优化技巧：
 
- 设置 reranker 的 max_seq_len，不要无限制传入超长 chunk

- 可以加简单缓存：同一个 query + 同一批候选，短时间不用重复 rerank

- 非关键路径，甚至可以做成「开关」：小知识库关闭 reranker；复杂知识库打开


上下文压缩
常见4大类实现方案（从简单到高级）
 
1）硬压缩：截断 / 滑动窗口（最弱，不叫真压缩）
 
- 做法：按字符/token直接砍掉超长部分

- 优点：零开销、最快

- 缺点：可能刚好把关键证据切掉；无法识别语义重要性

- 适用：应急，不适合作为主方案
 
坑：很多人以为“控制chunk大小”就是上下文压缩，那是预处理阶段；压缩是查询时在线做。
 
 
 
2）基于过滤器的压缩：元数据 / 相似度阈值过滤
 
- 思路：对检索到的每一条，看 embedding 相似度、reranker分数，低于阈值直接整条丢弃

- 代表： ContextualCompressionRetriever （LangChain里的基础版本）

- 优点：速度快、不调用LLM

- 缺点：只能丢整个chunk，不能“从chunk里抠出有用句子”；对“一段里一半有用一半垃圾”无能为力
 
 
 
3）LLM提取式压缩（最常用，也叫上下文萃取）
 
核心：让小模型 / 同一个LLM读一段文档 + 用户问题，只输出该段中能回答这个query的句子，其余丢掉

用户问题：{query}
下面这段文本里，哪些信息对回答上面问题有用？
只输出有用片段，不要解释，不要复述全部，没有用就输出【NONE】

文本：{chunk}

- 如果返回  NONE  → 直接丢弃这个chunk

- 如果返回片段 → 把片段加入最终上下文
 
✅ 优点：能做到句子级粒度，压缩率很高，有效缓解 lost‑in‑the‑middle
❌ 缺点：多一轮LLM调用 → 增加latency、增加token成本；小模型可能漏信息
👉 优化：用便宜高速小模型（如deepseek‑chat‑lite、qwen‑1.8b‑instruct）做压缩，不要用最贵主模型
 
注意：不是“让LLM把所有文档直接写成答案”，那就是提前回答，会丢失引用来源，不利于溯源！压缩只做「摘证据」，不做「生成最终答案」。
 
 
 
4）LLM重写式压缩（Aggregation / 上下文摘要）
 
- 做法：把多个相关chunk，基于query，改写浓缩成精简摘要

- 变种：

- 单chunk摘要：每块各自缩

- 多chunk聚合摘要：把一批chunk合并提炼

- 优点：token降得最多，上下文非常干净

- ⚠️ 重大风险：幻觉、事实偏移、丢失细节、丢失原文可溯源

- 适用：背景类、描述类长文档；不适合需要精确数字、条款、代码的场景
 
生产建议：尽量少用“完全重写”，优先「提取而不是改写」。


两个经典工程实现：LangChain ContextCompressionRetriever
原理包装链：
 BaseRetriever → 拿到N个doc → Compressor逐个处理doc → 返回压缩后的doc列表 → 送入最终LLM 
 
有两种内置 compressor：
 
-  LLMContextCompressor ：就是上面的「提取有用片段，NONE则丢弃」（推荐）

-  EmbeddingsFilter ：基于相似度过滤，无LLM，轻量
 
⚠️ 常见误区：不要把压缩放在文档入库预处理；压缩的价值在于「根据当前这一条query动态压缩」——同一个chunk，对A问题有用，对B问题就是垃圾，静态提前压缩做不到这点。


和 Reranker、摘要、Chunk分割 的清晰区分（必背）
 
- Chunk分割：离线预处理，把长文档切成小块，发生在入库时

- Embedding检索 / Hybrid检索：根据query，召回一批候选chunk

- Reranker：重打分+重排序+丢弃整条，粒度是文档块

- Context Compression：可以进到块里面，挑有用句子，剔除无关句子；是「检索后、生成前」的在线提纯

- 最终摘要：拿到全部证据后，生成给用户看的答案
 
完整高级RAG链路参考：
 
原始文档 → 加载 → 分割 → 清洗 → embedding → 存入向量库
用户query → query改写/扩展 → hybrid检索 → reranker重排 → 上下文压缩 → 加入prompt → LLM生成答案 + 引用溯源



什么时候一定要用上上下文压缩？
 
- 检索召回数 k 比较大（k>6‑8）

- chunk设置偏大（512‑1024 token）

- 很多chunk里面只有一两句话有用，其余都是背景

- 遇到大窗口但lost‑in‑the‑middle明显：前面、后面内容记得住，中间一堆证据被忽略

- 想在不换更大embedding、不换更强reranker的前提下，提升答案质量 + 降token消耗
 
❌ 不适合强行用：
 
- 本身k很小、chunk很短，再压缩收益极低，反而多耗时

- 需要一字不差引用法律、合同、代码：提取式压缩有可能切掉关键限定词


生产避坑清单
1. 压缩 ≠ 直接生成答案！一旦提前回答，就失去原文依据，幻觉风险暴涨

2. 尽量用「提取 > 重写摘要」；能不改写原文就不改写

3. 做压缩的模型不必很强，但要足够听话、严格按指令只输出片段；很多自由型大模型喜欢“额外发挥”

4. 必须处理  NONE ：不要把“没有价值的片段”也塞进去

5. 压缩不要过度：压缩率＞80%时，很容易漏关键细节（召回率牺牲）

6. 不要用压缩代替 reranker：最佳搭档是 reranker先粗筛，压缩再细榨

7. 记得保留  metadata （来源页码、文件名），压缩只改 page_content，不能丢掉溯源信息



- 上下文压缩 = 检索后、生成前，按当前query动态提纯证据

- reranker：操作「整块」，改顺序、丢整块

- 压缩：可以钻进块里，只留下对这个问题有用的句子

- 三类主流：相似度过滤、LLM提取、LLM摘要重写

- 黄金实践：reranker先筛，轻量模型做提取式压缩，慎用重度摘要

- 最大风险：过度压缩丢事实、压缩时提前生成答案、丢失溯源



Vision模型
什么是 LLM‑Vision（视觉大模型）
不是传统CV（YOLO、ResNet那种专门做检测/分类）。
本质：给文本大模型加上「图像→token序列」的编码器，让模型可以“读图片”，再用语言头输出文字。
 
- 它不是真正“看见像素”：不能直接理解像素；图像被转成一组视觉token，和文本token放在同一个上下文窗口里一起处理

- 仍然复用 Transformer 自注意力；推理逻辑还是LLM自回归

- 能力：看图描述、OCR、图表解读、推理、理解截图、简单空间关系；但不是高精度图像检测/分割模型
 
⚠️ 误区：Vision ≠ 万能识图；它擅长理解内容语义，不擅长精准像素级坐标、计数超多物体、超高分辨率细节。
 
注意：GPT‑4V、DeepSeek‑Vision、Qwen‑VL、GLM‑4V 都属于这一类「LLM系多模态」，和独立CV模型路线不一样。


核心内部流程
1. 图像预处理：图片缩放、切块（patch），统一到模型预期尺寸

2. 视觉编码器（Vision Encoder）：把每一块 patch 映射成固定维度的视觉embedding

3. 投影层（Projection）：把视觉向量「对齐到LLM的词向量空间」——这是最关键！让文本Transformer能读懂图像向量

4. 插入上下文：视觉token和普通文本token拼接，一起送入Transformer

5. LLM解码器：像普通对话一样，自回归输出文本
 
👉 重点：没有独立“检测头/分割头”；它靠语言去“描述看到了什么”，不是显式做目标检测。
 
和早期CLIP区别：
CLIP：图像‑文本相似度，只做向量匹配，不能生成语言
GPT‑4V / DeepSeek‑Vision：可以基于图像做连续推理、多轮对话



