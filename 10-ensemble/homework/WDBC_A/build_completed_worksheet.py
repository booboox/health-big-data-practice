"""Assemble all worksheet answers and verified experimental figures."""
from pathlib import Path
import re, json
import pandas as pd
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE=Path(__file__).resolve().parent; FIG=BASE/'工作单配图'
doc=Document();sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.8);sec.bottom_margin=Cm(1.7);sec.left_margin=Cm(1.8);sec.right_margin=Cm(1.8)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    s=doc.styles[name];s.font.name='Arial';s.font.color.rgb=RGBColor(0,0,0)
    s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    s.font.size=Pt(10 if name=='Normal' else 9 if name=='Caption' else 15 if name=='Heading 1' else 11 if name=='Heading 2' else 22 if name=='Title' else 11)
    s.paragraph_format.space_after=Pt(6);s.paragraph_format.line_spacing=1.15
doc.styles['Normal'].paragraph_format.widow_control=True
for style in doc.styles:
    for border in list(style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
doc.styles['Caption'].font.bold=False
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');r._r.addnext(field)
doc.core_properties.title='WDBC 集成学习与 Bootstrap 完整工作单'
doc.core_properties.author='';doc.core_properties.subject='第10讲学生工作单 A至H 实验与回答'
md=[];page_count=0
def page(title):
    global page_count
    if page_count:doc.add_page_break()
    page_count+=1;doc.add_heading(title,level=1);md.append('\n## '+title+'\n')
def p(text):
    doc.add_paragraph(text);md.append(text+'\n')
def q(title,text):
    doc.add_heading(title,level=2);md.append('\n### '+title+'\n');p(text)
def table(headers,rows,widths=None):
    tb=doc.add_table(rows=1,cols=len(headers));tb.alignment=WD_TABLE_ALIGNMENT.CENTER;tb.autofit=False
    widths=widths or [17.4/len(headers)]*len(headers)
    for c,w in zip(tb.columns,widths):c.width=Cm(w)
    for c,s in zip(tb.rows[0].cells,headers):c.text=str(s)
    for row in rows:
        for c,s in zip(tb.add_row().cells,row):c.text=str(s)
    for i,row in enumerate(tb.rows):
        trPr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trPr.append(cant)
        if i==0:
            repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
        for j,c in enumerate(row.cells):
            c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=c._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
            cp.append(borders)
            mar=OxmlElement('w:tcMar')
            for side in ['top','left','bottom','right']:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),'85');el.set(qn('w:type'),'dxa');mar.append(el)
            cp.append(mar)
            if i==0 or i%2:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EAF1F5' if i==0 else 'F7F9FA');cp.append(shade)
            for par in c.paragraphs:
                par.paragraph_format.space_after=Pt(2);par.paragraph_format.line_spacing=1.1
                if j>0 and len(str(c.text))<26:par.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in par.runs:run.font.size=Pt(9);run.bold=i==0
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
    md.extend('| '+' | '.join(map(str,row))+' |' for row in rows);md.append('')
def image(name,caption,width=17.1):
    path=FIG/name;assert path.exists()
    par=doc.add_paragraph();par.paragraph_format.keep_with_next=True
    par.alignment=WD_ALIGN_PARAGRAPH.CENTER;par.add_run().add_picture(str(path),width=Cm(width))
    pic=par.runs[0]._r.xpath('.//wp:docPr')[0];pic.set('descr',caption)
    doc.add_paragraph(caption,style='Caption')
    md.append(f'![{caption}]({path.as_posix()})\n')

doc.add_paragraph('WDBC 集成学习与 Bootstrap',style='Title')
page('数据与协议')
p('姓名：________    小组：________    日期：________')
p('研究问题：在相同数据与评价协议下，模型聚合能否改善分类表现与预测稳定性？本实验采用 WDBC 数据，完成工作单 A 至 H。集成改善了课程基线单树，但没有超过逻辑回归和 RBF-SVM；模型建议仅限于当前开发数据。')
q('1 记录 特征与标签','每行是一个乳腺肿块细针穿刺图像提取的细胞核形态特征记录，不是一颗细胞。30 个数值特征涵盖 10 类形态量的 mean、standard error、worst 汇总。sample_id 仅用于追踪，不输入模型，以免学习无意义的编号关联。malignant=1 为恶性，0 为良性，是当前数据中的诊断标签，不能解释为未来患病概率。')
table(['集合','样本数','良性','恶性','缺失值'],[['训练集',341,'214（62.76%）','127（37.24%）',0],['验证集',114,'71（62.28%）','43（37.72%）',0],['锁定测试集',114,'未读取','未读取','未检查']], [3,2,4.5,4.5,3.4])
p('训练与验证各 32 列：编号、30 个预测特征、标签。两份表列名及顺序一致；样本编号各自无重复、彼此无重叠。')
q('2 训练 验证与测试','训练集用于拟合参数；验证集用于开发阶段比较模型或选择超参数；测试集在方案锁定后用于最终评价。提前查看测试标签并据其改模型会污染最终评价。本次未读取测试标签。F 的开发集交叉验证按课程安排使用 train+validation 共 455 条记录，与固定验证结果共享数据，不能视为独立重复验证。')
p('实验设置：random_state=1010，sklearn 1.9.1。LR/SVM 使用折内标准化，树模型不标准化。只使用预设参数作公平比较，E3 的学习率探索单独报告。患者分组与时间字段不可用，无法核验患者独立性。')

page('A 决策树的预测与深度')
q('A1 示例树预测','按课件第 5 页：worst perimeter=120>105.95，沿右支；mean concave points=0.08>0.05，再沿右支，到达良性 3、恶性 112 的叶节点。因此预测恶性，概率为 112/(112+3)=0.9739，约 0.97。叶节点概率来自到达该叶的训练记录中各类别的比例，不是额外的临床风险估计。')
q('A2 节点怎样选择切分','节点比较候选特征和阈值，选择加权子节点不纯度下降最大的切分。Gini=1−Σp_k²；越小表示类别越集中，可解释为从当前类别分布独立抽取两次时类别不同的概率。它衡量混杂程度，不是分类准确率。')
q('A3 不同深度的训练与验证表现','仅在原始训练集拟合，除深度外参数相同。验证准确率在深度 3 最高，深度不限时训练全对而验证未进一步改善，呈现过拟合迹象。深度 3 与 4 的准确率仅差一个验证样本。')
d=pd.read_csv(BASE/'tree_depth_results.csv')
table(['最大深度','训练准确率','验证准确率','验证 BA','验证 AUC'],[[r.max_depth,f'{r.train_accuracy:.2%}',f'{r.validation_accuracy:.2%}',f'{r.validation_balanced_accuracy:.4f}',f'{r.validation_roc_auc:.4f}'] for r in d.itertuples()],[3,3.6,3.6,3.6,3.6])
image('Figure_A3_tree_depth.png','图 1  固定划分上的决策树深度实验。训练 n=341，验证 n=114；阴影标出深度 3。None 为不限深度，实际深度 7。点为单次结果，未估计置信区间。')

page('A 训练扰动与标准化')
q('A4 Bootstrap 后根节点是否相同','从 341 条训练记录中有放回抽取 341 次，重复 100 轮，各拟合深度 2 的树，模型种子保持固定。根节点出现 6 种特征；同一特征的阈值也变化，说明树的结构对训练样本扰动敏感。结构变化不自动等于预测不稳定，后续 D/G 用固定验证记录的预测变化直接检验。')
image('Figure_A4_bootstrap_roots.png','图 2  100 次训练重抽样的根节点频数和阈值范围。频数与百分比数值相同；阈值保留各特征原始单位，不能跨特征直接比较。')
p('根节点频数：worst perimeter 44 次，worst area 21 次，worst concave points 16 次，worst radius 11 次，mean concave points 7 次，mean area 1 次。worst perimeter 的切分阈值约为 103.25–115.60。')
q('A5 为什么树通常不需要标准化','树按单个特征与阈值的大小关系划分。正比例换单位不改变排序，通常只改变阈值表示，所以一般无需标准化。带正则化的逻辑回归会受特征尺度对系数惩罚的影响，标准化也有利于优化；RBF-SVM 的核依赖距离，尺度大的特征可能主导距离。标准化参数只在训练数据上拟合，交叉验证时也必须在各折训练部分拟合。浮点和并列切分可能导致树的微小数值差异。')

page('B 硬投票与软投票')
p('硬投票对类别取多数；软投票对类别 1 的概率取平均，再以 0.5 为阈值。以下是工作单给定示例，不是 WDBC 模型输出。')
table(['样本','真值','类别预测','概率预测','硬投票','平均概率','软投票'],[['A',1,'1 1 0','.80 .70 .45',1,'0.6500',1],['B',0,'0 1 1','.20 .60 .65',1,'0.4833',0],['C',1,'0 1 1','.35 .75 .80',1,'0.6333',1],['D',0,'0 0 1','.15 .25 .70',0,'0.3667',0]],[1.2,1.2,2.8,4,2,3,3.2])
image('Figure_B1_hard_soft_voting.png','图 3  三模型概率与聚合类别。菱形为概率平均，竖虚线为分类阈值；横带突出样本 B。纵向错开仅用于辨认各模型点。')
p('样本 B 的硬投票以 2∶1 预测为类别 1，发生错误；软投票的平均概率为 (0.20+0.60+0.65)/3=0.4833，低于 0.5，预测为类别 0，与真值一致。概率平均保留了模型预测强度的信息，但此例不能证明软投票普遍更优。')

page('B 集成方差的理论与验证')
p('在各模型误差方差均为 σ²、两两相关系数均为 ρ 的设定下，平均后的方差为 ρσ²+(1−ρ)σ²/B。')
q('B1 模型数量增加的极限','B 趋于无穷时，第二项趋于零，方差趋于 ρσ²。模型具有共享误差时，仅增加数量不能消除所有波动。')
q('B2 模型总犯相同错误时','ρ 接近 1 时，集成方差接近单模型方差，平均帮助有限。除数量外，还需要具有有效差异的基学习器。')
q('B3 平均与偏差','简单平均不会自动消除模型共有的系统性偏差。Bagging 主要用于降低方差，因此适合拟合能力强、对样本扰动敏感的深树；通常更稳定的逻辑回归收益可能较小。真实 Bagging 的偏差变化仍取决于算法和数据，不能断言始终完全不变。')
image('Figure_B2_ensemble_variance.png','图 4  构造误差 e_b=√ρ Z₀+√(1−ρ)Z_b，单模型方差为 1。每设置模拟 30,000 次，曲线为理论，点为模拟。此为公式机制验证，不是 WDBC 性能实验。')
p('B=100、ρ=0/0.5/1 时，理论方差为 0.010/0.505/1.000，模拟为 0.0101/0.5068/1.0022，有限模拟下与理论接近。')

page('C Bootstrap 抽样与模拟')
q('C1 重复记录与袋外记录','一次抽样为 [2,2,5,1,2]。样本 2 出现 3 次；不同样本为 {1,2,5}，占 60%；样本 3、4 未被抽到，为袋外 OOB 样本，占 40%。抽取 5 次不等于每条记录都出现一次。')
q('C2 一千次模拟','分别从 n=5 和 n=341 个索引中有放回抽取 n 次，重复 1,000 轮，seed=1010。由于重复抽中的记录数量随机，不同样本占比与袋外比例会随轮次变化。')
image('Figure_C12_bootstrap_sampling.png','图 5  有放回抽样示意与不同样本占比分布。箭头表示记录对应关系。虚线为有限样本精确期望；n=341 对应 WDBC 训练规模，模拟仅使用索引。')
table(['样本数','模拟不同样本占比','模拟袋外比例','理论不同样本占比'],[[5,'67.50%','32.50%','67.232%'],[341,'63.1965%','36.8035%','63.2661%']],[3,5,4.4,5])
p('精确期望为 1−(1−1/n)^n；袋外比例期望为 (1−1/n)^n。大样本时分别趋近 63.2% 和 36.8%，不能将渐近值当作 n=5 的精确期望。')

page('C Bagging 与袋外评价')
q('C3 多训练集与并行','Bagging 通过多次有放回抽样生成多份训练集，各自拟合基学习器后聚合预测；因为模型之间没有训练顺序依赖，可以并行。训练集之间允许重叠，模型误差也未必独立。本次代码采用 n_jobs=1，没有测量并行加速。')
q('C4 OOB 的计算和用途','对每条训练记录，只聚合未抽到它的树的预测，再与真值比较计算 OOB 误差。OOB 可辅助内部评价和选参，减少另留验证数据的需求；反复据其选参也会引入选择偏差，不能替代锁定测试或外部验证。分组数据按行 OOB 可能泄漏同组信息。')
image('Figure_C34_bagging_oob.png','图 6  独立训练和记录特定的袋外预测。实测 100 棵全特征深树，各训练记录对应 21–52 棵袋外树，平均 36.8 棵。')
table(['评价','准确率','错误率','BA','ROC-AUC'],[['OOB','94.43%','5.57%','0.9380','0.9826'],['独立验证','98.25%','1.75%','0.9813','0.9984']],[3.4,3.5,3.5,3.5,3.5])
p('两种评价涉及不同记录、样本数和预测树数，不要求数值相等，不能据此断言 OOB 总偏低。')

page('D 单树与集成的表现和稳定性')
p('单树沿用课程深度 4 基线；Bagging 为 100 棵全特征深树，RF 为 100 棵每节点 sqrt 特征深树。稳定性实验使用共享的 30 份训练重抽样及固定验证记录，算法种子不变。')
dt=pd.read_csv(FIG/'Figure_D_model_comparison.csv').merge(pd.read_csv(FIG/'Figure_D_prediction_stability.csv'),on='model')
table(['模型','验证 AUC','验证 BA','平均预测 SD','类别翻转率'],[[r.model,f'{r.roc_auc:.4f}',f'{r.balanced_accuracy:.4f}',f'{r.mean_prediction_sd:.4f}',f'{r.class_flip_rate:.2%}'] for r in dt.itertuples()],[4.2,3.3,3.3,3.3,3.3])
image('Figure_D1_performance_stability.png','图 7  验证判别表现及 30 次共享重训的预测波动。SD 为每记录恶性概率波动后取平均；翻转率为曾落在 0.5 两侧的记录比例。图示不是置信区间。')
p('观察机制：单树对训练扰动敏感；Bagging 通过平均多树降低整体波动；RF 再增加候选特征随机性，本次平均 SD 和翻转率最低。Bagging 的验证 AUC 和 BA 最高，RF 并未在所有性能指标上领先。单树与基树深度不同，因此不能将所有差异完全归因为聚合。A 的深度 3 单树仍保留为候选。')

page('D 特征随机化与树数')
q('D1 特征随机化的作用','每个节点仅考虑随机抽取的部分特征，可以减少不同树对同一强特征的依赖，从而可能降低误差相关性。若特征高度冗余，其他特征可相互替代，收益取决于数据。若树总犯相似错误，增加数量的收益受共享误差限制，不保证持续改善。本实验的预测稳定性不能当作直接测得的误差相关系数。')
q('D2 增加树数是否持续改善','从 1 棵增加到 10 棵时验证 AUC 明显提高；10–50 棵之后基本进入平台。有限验证集上并不严格单调。随机森林增加树数通常使平均预测更稳定，而不是像增加单树深度那样持续提高复杂度，但不能修复偏差或不当参数，验证结果仍可能小幅下降。')
image('Figure_D2_tree_count.png','图 8  相同划分与种子下，1、10、50、100、300 棵树的验证表现。横轴为对数刻度，连线连接实测设置；未进行独立重复种子试验。')
c=pd.read_csv(FIG/'Figure_D_tree_count_curve.csv')
table(['树数','Bagging AUC','RF AUC'],[[n,f'{c[(c.model=="Bagging")&(c.n_trees==n)].iloc[0].roc_auc:.4f}',f'{c[(c.model=="Random forest")&(c.n_trees==n)].iloc[0].roc_auc:.4f}'] for n in [1,10,50,100,300]],[3.4,7,7])

page('D 两种特征重要性')
q('D3 排名是否一致','不完全一致。不纯度前三名是 worst area、worst perimeter、worst concave points；置换前三名是 worst area、worst concave points、worst texture。例如 worst perimeter 的不纯度排名第 2，置换排名第 16，平均置换 AUC 下降约 −0.00004。')
image('Figure_D3_importance_comparison.png','图 9  同一个 100 棵 RF 的不纯度重要性与验证置换重要性。各显示前十名，单位不同。右图为 50 次打乱的均值 ±1 SD，非置信区间。')
image('Figure_D3_size_correlation.png','图 10  训练集尺寸特征相关矩阵，组内相关约 0.94–1.00。显示两位小数的 1.00 不代表精确完全相关。完整 30 特征矩阵另存于配图文件夹。',width=9)
p('训练集共 21 对特征满足 |r|>0.9。不纯度重要性可能在相关特征间分摊或偏向某个替代指标；单独置换一个特征时，其他相关特征可能替补，导致重要性偏低。不纯度重要性一般偏向候选切分较多的特征；置换还可能破坏原联合分布。接近零或略负不等于无用或保护效应，重要性不是因果效应。')

page('E AdaBoost 的顺序依赖')
q('E1 为什么提高错分样本权重','提高错分权重后，下一轮学习器在加权损失中更关注这些记录。第一轮错分 23 条；归一化后错分者权重变为原来 7.413 倍，正确者为 0.536 倍。下一轮依赖这些更新权重，因此具有顺序依赖。Bagging 的重抽样不依赖上一模型的错误。噪声或错标也可能被强化，不能保证泛化提高。')
image('Figure_E1_adaboost_weight_updates.png','图 11  显式二分类离散 AdaBoost 的 8 轮深度 1 树。左侧是各轮拟合前权重，右侧是本轮归一化权重变化倍数。训练记录按编号排序。')
p('实现定义：标签和预测用 ±1 编码，ε 为加权错误率，α=0.5 ln[(1−ε)/ε]；更新 w_i←w_i exp(−α y_i h_i)，再归一化。各轮模型种子为 1010。本实验显式实现该形式以追踪权重，不将数值当作所有 AdaBoost 软件变体的默认输出。')
p('顺序依赖来自“当前模型产生下一轮训练权重”。这与 Bagging 的独立基树训练不同，但不排除每轮内部运算的并行优化。')

page('E 梯度提升拟合负梯度')
q('E2 新树拟合什么方向','新树拟合当前损失关于原始分数的负梯度。二分类 log-loss 下为 y−p：y=1 时 1−p 为正，希望提高恶性分数；y=0 时 −p 为负，希望降低分数。拟合伪残差后，更新叶节点并按学习率叠加。下一轮目标取决于当前累积模型，不能独立并行拟合所有轮次。')
image('Figure_E2_gradient_boosting_residuals.png','图 12  WDBC 训练集梯度提升完成前 5 轮后的概率与负梯度。learning_rate=0.1，max_depth=3。图中部分点重合；横线和竖虚线分别表示零负梯度及 0.5 分类阈值。')
p('正确分类仍可贡献梯度：本次 336 个正确分类样本的平均 |y−p| 为 0.283，5 个错分样本为 0.649。因此梯度提升不是只拟合错分样本。有限深度树只是近似拟合梯度，不能保证逐个样本精确获得希望的更新幅度。这里是训练诊断，不是验证性能估计。')

page('E 学习率与训练轮数')
q('E3 轮数过多会发生什么','预设 learning_rate=0.03、0.1、0.3，分别拟合 500 轮深度 3 树。训练损失持续下降，验证损失先降后升，呈现过拟合迹象；验证 AUC 却仍接近 0.997，说明排序指标稳定时，错误预测仍可能变得过于自信。轮数应按预设验证指标或早停规则决定，不能看训练损失或测试结果。')
image('Figure_E3_boosting_learning_curves.png','图 13  三种学习率的逐轮训练与验证曲线。虚线标出各自验证 log-loss 最小值。所有轮次拟合完成后检查曲线，未启用自动早停；无置信区间。')
table(['学习率','最小损失轮数','最小验证损失','500 轮验证损失'],[['0.03',302,'0.0664','0.0800'],['0.10',83,'0.0733','0.1617'],['0.30',36,'0.0791','0.1599']],[3,4.5,5,4.9])
p('较小学习率在本次需要更多轮。按验证损失暂优为 0.03/302 轮；这是开发数据选参结果，不是无偏最终表现。F 仍使用课程预设的 0.1/100 轮作比较，以区分探索和基线。')

page('E 梯度提升树的不同实现')
q('E4 XGBoost LightGBM 与 sklearn 的关系','三者共享逐轮叠加树、改善损失的思想。sklearn 传统 GradientBoostingClassifier 拟合负梯度，并用学习率、轮数、深度、子采样控制模型。XGBoost 结合梯度和 Hessian，加入树或叶权重复杂度正则，并优化稀疏数据与分裂搜索。LightGBM 采用直方图分裂、leaf-wise 最佳叶优先生长；原始论文提出 GOSS 梯度采样和 EFB 互斥特征捆绑。GOSS 是可选策略，不是默认必启用。参见文末来源 1 至 6。')
image('Figure_E4_boosting_family.png','图 14  梯度提升树的共同框架及代表性实现特性。为文献归纳而非 WDBC 性能排名；箭头不表示软件继承或发展顺序。')
p('sklearn 另有支持缺失值的直方图 HistGradientBoosting，不能概括成 sklearn 不支持缺失值或没有正则化。各特性也并非某库独占。Leaf-wise 在小数据上可能过拟合，仍需约束复杂度。库内部并行优化不等于所有 boosting 轮次独立。按题目要求，本题查阅资料，无需安装运行。')

page('F 公平比较与选型')
p('四模型使用相同的固定划分，以及 455 条开发记录上共享的分层五折。LR/SVM 的 StandardScaler 在管道内按折训练；SVM 以 decision_function 计算 AUC，不当作概率。预设 LR 默认 C=1，SVM C=1/RBF，RF 100 棵/sqrt，GB 0.1/100 轮/深度 3。')
ft=pd.read_csv(FIG/'Figure_F_model_comparison.csv')
table(['模型','验证 AUC','验证 BA','CV AUC 均值 ± SD','解释成本'],[[r.model,f'{r.validation_roc_auc:.4f}',f'{r.validation_balanced_accuracy:.4f}',f'{r.cv_auc_mean:.4f} ± {r.cv_auc_sd:.4f}','较低' if r.model=='Logistic regression' else '较高'] for r in ft.itertuples()],[4.3,2.4,2.4,5.8,2.5])
image('Figure_F_fair_model_comparison.png','图 15  固定验证表现、共享五折 AUC 与验证 ROC。黑色标记及误差线为均值 ± 折间 SD，不是置信区间；AUC=1 的曲线部分重合。')
q('F1 集成是否胜过基线','没有。LR 的 CV AUC 均值比 RF 高 0.0049，比 GB 高 0.0059，差值小于约 0.01 的折间波动。但 SD 不是显著性检验，不能用误差线重叠与否判定优势。验证集 SVM 全对不等于未来错误率为零。')
q('F2 优先选择与改变条件','按课件第 14 页的选型思路，优先 LR 并保留 RBF-SVM：本次 LR 的 CV AUC 和 CV BA 均值较好，解释成本相对较低。若后续独立数据在预设指标上显示集成有足够大且可重复的收益，解释和计算成本也可接受，则改选。当前稳定性比较未覆盖 LR/SVM，不能推断 RF 比它们更稳定。')

page('G 稳定性与不确定性的范围')
p('复用 D 的 30 次共享重抽样结果，不新增重训。固定验证记录和算法种子时，预测 SD 主要刻画训练样本扰动敏感性。平均 SD 为单树 0.0855、Bagging 0.0415、RF 0.0284，翻转率为 24.56%、15.79%、7.02%。')
image('Figure_G_stability_uncertainty.png','图 16  逐样本 SD 的经验累积分布、配对比较，以及三类不确定性。右上点低于对角线表示该记录的集成 SD 更低；重合点可能相互遮挡。')
p('平均降低不等于每个样本都降低：Bagging 在 64/114 条记录的 SD 低于单树，RF 为 50/114。单树存在部分较大跳变，拉高平均值；其中位 SD 却仅 0.00545，低于 Bagging 0.01055 和 RF 0.01400。零波动也不保证正确。')
p('固定模型的指标区间需要固定模型、对评价记录重抽样，本次未计算；它不包含重新训练和重新选模型的波动。外部人群不确定性涉及新医院、时间和人群，需要独立外部数据。当前内部稳定性实验不能得到个体临床风险区间，也不能建立跨医院有效性。')

page('H Bagging 与 Boosting 对照')
table(['方面','Bagging 含随机森林','Boosting'],[
    ['训练方式','基学习器可独立并行训练','轮次顺序依赖；每轮内部运算可并行'],
    ['每轮数据或权重','各用一份 Bootstrap 样本；RF 另随机选候选特征','AdaBoost 更新样本权重；GB 根据当前预测拟合负梯度'],
    ['典型基学习器','通常用深树，拟合能力强但方差较高','常用浅树；AdaBoost 常用树桩，不是硬性限制'],
    ['偏差与方差','主要降低方差；不能自动消除共同偏差','逐步改善拟合，常降低偏差；也可能影响方差'],
    ['噪声与轮数','通常较稳健但不免疫噪声；加树趋于稳定，收益递减','可能持续关注噪声；轮数过多可过拟合，需验证或早停'],
    ['关键超参数','n_estimators、max_features、max_depth、min_samples_leaf、max_samples','learning_rate 与 n_estimators 联动；max_depth、叶节点约束及正则化']
],[3.1,7.15,7.15])
p('实验对应：D 中集成的平均 SD 低于单树，支持方差改善的解释；E 中 AdaBoost 权重依赖前轮，GB 用 y−p 更新，且验证损失会随轮数过多而回升。两类机制并不意味着任何数据上都有固定的性能排名。')
memo=('在本次WDBC数据和预设协议下，我优先选择逻辑回归，并保留RBF支持向量机。逻辑回归的开发集五折AUC为0.9942±0.0099，随机森林为0.9893±0.0100，尚无证据表明集成具有稳定优势。共享训练重抽样显示，随机森林的平均预测标准差为0.0284，低于单树的0.0855，但未直接比较逻辑回归的重训稳定性，且并非每个样本都更稳定。逻辑回归解释成本相对较低，相关特征仍会影响系数解释；特征重要性不能视为因果效应。当前结果限于开发数据，固定验证与交叉验证并非独立证据，锁定测试集尚未评价，也不能推断跨医院表现或临床效用。若独立外部队列在预设指标上显示集成具有可重复且足够大的收益，我会重新考虑模型选择。')
q('H2 研究备忘录',memo)
memo_count=len(re.sub(r'\s','',memo));assert 250<=memo_count<=400,memo_count
(BASE/'H_必做对照与研究备忘录.md').write_text('\n'.join(md[md.index('\n## H Bagging 与 Boosting 对照\n'):]),encoding='utf-8')

page('参考资料与复现索引')
refs=[
    '1. scikit-learn. GradientBoostingClassifier 官方文档。https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html',
    '2. scikit-learn. Ensemble 用户指南，含直方图梯度提升。https://scikit-learn.org/stable/modules/ensemble.html',
    '3. XGBoost. Introduction to Boosted Trees。https://xgboost.readthedocs.io/en/stable/tutorials/model.html',
    '4. Chen T, Guestrin C. XGBoost A Scalable Tree Boosting System. KDD, 2016. https://doi.org/10.1145/2939672.2939785',
    '5. LightGBM. Features 与 Parameters 官方文档。https://lightgbm.readthedocs.io/en/latest/Features.html ; https://lightgbm.readthedocs.io/en/latest/Parameters.html#data_sample_strategy',
    '6. Ke G, et al. LightGBM A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30, 2017. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html',
    '7. WDBC 数据来源与课程划分见 data/README.md。UCI 数据页：https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic',
    '8. 课程来源为 student-worksheet.md 和 slides/ensemble-pbl.pptx；示例树对应第 5 页，选型对应第 14 页。在线资料核对日期为 2026-11-16。'
]
for s in refs:p(s)
doc.add_heading('代码与数据',level=2)
p('所有代码位于 WDBC_A 文件夹。图像、中文图注、逐轮指标、样本预测与元数据位于其下“工作单配图”。所有结果均为实际计算或明确标注的示例及机制模拟。')
table(['任务','复现脚本'],[['A3','tree_depth_comparison.py'],['A4 与 A 配图','plot_worksheet_a.py'],['B1 与 B 方差','plot_worksheet_b.py / plot_variance_simulation.py'],['C','plot_bootstrap_sampling.py / plot_bagging_oob.py'],['D','experiment_d_ensembles.py / experiment_d3_importance.py'],['E','experiment_e12_boosting.py / experiment_e3_learning_curves.py / plot_e4_boosting_family.py'],['F 与 G','experiment_f_fair_comparison.py / plot_g_uncertainty.py'],['完整工作单','build_completed_worksheet.py']],[3.1,14.3])
p('环境：Python 3.12，scikit-learn 1.9.1，numpy、pandas、matplotlib、python-docx。各实验元数据保留种子和版本。主比较未使用锁定测试标签；测试评价按课程安排在方案锁定后进行。')

output=BASE/'第10讲_WDBC完整工作单_已完成.docx';doc.save(output)
(BASE/'第10讲_WDBC完整工作单_已完成.md').write_text('\n'.join(md),encoding='utf-8')
(BASE/'完成检查.json').write_text(json.dumps({'pages_planned':page_count,'memo_characters_including_ascii_and_punctuation':memo_count,
    'coverage':['数据1','数据2','A1','A2','A3','A4','A5','B投票','B1','B2','B3','C1','C2','C3','C4','D主表','D1','D2','D3','E1','E2','E3','E4','F主表','F1','F2','G','H1','H2'],
    'figures_embedded':len(doc.inline_shapes),'locked_test_used':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(output);print('Planned pages:',page_count,'Memo characters:',memo_count,'Figures:',len(doc.inline_shapes))
