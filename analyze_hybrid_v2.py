"""Post-run audit and paired comparisons, without tuning or evaluating new models."""
import json
import hashlib
from pathlib import Path
import numpy as np
from train_hybrid_v2 import OUT, ROOT, read_rows, write_json, summarize


def paired(left,right):
    a={r['episode']:r for r in left};b={r['episode']:r for r in right}
    assert len(a)==len(left)==len(b)==len(right) and set(a)==set(b)
    ids=sorted(a)
    delta=np.array([int(b[i]['outcome']=='success')-int(a[i]['outcome']=='success') for i in ids])
    boot=np.random.RandomState(12345).choice(delta,(10000,len(ids))).mean(1)
    return dict(episodes=len(ids),difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        wins=[i for i,d in zip(ids,delta) if d==1],losses=[i for i,d in zip(ids,delta) if d==-1])


def analyze(out=OUT):
    out=Path(out);report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    old=ROOT/'hybrid_experiment_v1'
    groups={name:read_rows(out/(name+'.jsonl')) for name in (
        'base_original200','hybrid_original200','reference_holdout200','base_holdout200','hybrid_holdout200')}
    groups['reference_original200']=read_rows(old/'base_original200.jsonl')
    audit={name:summarize(rows) for name,rows in groups.items()}
    for dataset in ('original200','holdout200'):
        a=sorted(groups['reference_'+dataset],key=lambda r:r['episode'])
        b=sorted(groups['base_'+dataset],key=lambda r:r['episode'])
        audit['zero_command_identity_'+dataset]=all((x['outcome'],x['steps'])==(y['outcome'],y['steps'])
            and [t['executed'] for t in x['last20']]==[t['executed'] for t in y['last20']] for x,y in zip(a,b))
    for dataset in ('original200','holdout200'):
        for baseline in ('reference','base'):
            audit[baseline+'_vs_hybrid_'+dataset]=paired(groups[baseline+'_'+dataset],groups['hybrid_'+dataset])
    train_rows=read_rows(out/'train_episodes.jsonl')
    audit['training']={label:summarize([r for r in train_rows if r['sampling']==label]) for label in ('regular','intercept')}
    audit['train_route_choices']=np.array([r['route_choices'] for r in train_rows]).sum(0).tolist()
    audit['train_route_opportunities']=sum(r['route_opportunities'] for r in train_rows)
    left,right=audit['train_route_choices'][1:]
    audit['train_decision_route_choices']=[audit['train_route_opportunities']-left-right,left,right]
    audit['train_threat_steps']=sum(r['threat_steps'] for r in train_rows)
    validation=read_rows(out/'validation.jsonl');base_validation=report['base_validation']
    best=next(r for r in validation if r['step']==report['best_step'])
    audit['validation_improvement']=best['success_rate']-base_validation['success_rate']
    audit['validation_gate_met']=bool(best['success_rate']>=.75 and audit['validation_improvement']>=.05-1e-12
        and best['counts']['obstacle_collision']/best['episodes']<=.05)
    manifest=json.loads((out/'source_hashes.json').read_text())
    audit['experiment_sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    old_manifest=json.loads((old/'source_manifest.json').read_text())
    audit['v1_sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old_manifest.items())
    audit['selected_checkpoint_sha256']=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()
    commands=[t for r in groups['hybrid_holdout200'] for t in r['last20'] if t['prediction']['threat']>.05]
    audit['holdout_last20_active_commands']=dict(count=len(commands),
        mean_abs_heading_degrees=float(np.mean([abs(t['command'][0])*90 for t in commands])) if commands else 0,
        mean_speed_command=float(np.mean([t['command'][1] for t in commands])) if commands else 0)
    write_json(out/'analysis.json',audit)
    lines=['# v2 配对评估与失败分析','',f"训练完成 {report['completed_steps']} 步，最佳验证检查点 {report['best_step']} 步。",'',
        '| 数据集 | 控制器 | 成功 | 被捕 | 撞障碍 | 超时 | 平均换路次数 |','|---|---|---:|---:|---:|---:|---:|']
    for dataset,label in (('original200','原200局'),('holdout200','新留出200局')):
        for method,name in (('reference','v1基础'),('base','v2基础'),('hybrid','v2基础＋PPO')):
            r=audit[method+'_'+dataset];c=r['counts']
            lines.append(f"| {label} | {name} | {c['success']}/{r['episodes']}（{r['success_rate']:.1%}） | {c['caught']} | {c['obstacle_collision']} | {c['timeout']} | {r['avg_switches']:.2f} |")
    for dataset in ('original200','holdout200'):
        for baseline in ('reference','base'):
            p=audit[baseline+'_vs_hybrid_'+dataset]
            lines+=['',f"{dataset} 相对 {baseline}：成功率差 {p['difference']:+.1%}；配对胜/负 {len(p['wins'])}/{len(p['losses'])}；95%配对 bootstrap 区间 [{p['ci95'][0]:+.1%}, {p['ci95'][1]:+.1%}]。"]
    lines+=['',f"验证集相对 v2 基础提升 {audit['validation_improvement']:+.1%}；原定验证门槛通过：{audit['validation_gate_met']}。没有自动延长训练。",'',
        '最终v2基础不降低巡航速度，零指令应与原强基础控制器一致。曾使成功率下降的8%速度余量方案已取消，记录另存。',
        '原200局是已知对照集；新留出集未参与参数选择。配对区间只反映这些场景的抽样差异，不代表跨训练种子的稳定性。',
        '下面列出新留出集中相对原基础控制的得失场景；完整失败前20步见 failure_traces.jsonl。','']
    p=audit['reference_vs_hybrid_holdout200']
    lines += [f"- 新增成功场景：{p['wins']}",f"- 丢失成功场景：{p['losses']}",
        f"- 仅在有效路线决策时计数（保留、左、右）：{audit['train_decision_route_choices']}；路线决策机会：{audit['train_route_opportunities']}。",
        f"- 留出集最后20步中有效偏转的平均绝对角度：{audit['holdout_last20_active_commands']['mean_abs_heading_degrees']:.2f}°；这不是整局平均。",
        f"- v1原文件未变：{audit['v1_sources_unchanged']}；本次实验源码未变：{audit['experiment_sources_unchanged']}。"]
    lines+=['','## 可观测的学习瓶颈','']
    for label in ('regular','intercept'):
        r=audit['training'][label]
        lines.append(f"- 探索训练的 {label} 场景：{r['counts']['success']}/{r['episodes']} 成功（{r['success_rate']:.1%}），这些不是确定性评估成绩。")
    total_switches=sum(r['switches'] for r in groups['hybrid_original200']+groups['hybrid_holdout200'])
    lines.append(f"- 选定模型在两组200局确定性评估共切换路线 {total_switches} 次。若为零，评估差异来自连续偏转/速度恢复指令，而非学会了换路。")
    p=audit['reference_vs_hybrid_original200']
    lines.append(f"- 原200局相对原基础新增成功 {p['wins']}，丢失成功 {p['losses']}。")
    lines+=['- 解释边界：增加探索和难例并不保证学会绕行。困难样本若正反馈很少，下一步应先核查可解性与课程难度；本轮结果不能证明具体哪一项调整单独有效，需后续消融。',
        '- 运行建议：未达到预定增益门槛时，继续使用原基础控制器。v2模型保留为实验结果，不自动替换，也不自动追加训练。']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (out/'failure_traces.jsonl').open('w',encoding='utf-8') as f:
        for dataset in ('original200','holdout200'):
            for method in ('reference','base','hybrid'):
                for row in groups[method+'_'+dataset]:
                    if row['outcome']!='success':f.write(json.dumps(dict(dataset=dataset,method=method,**row),ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if 'vs_hybrid' in k or k.endswith('unchanged')},indent=2))
    return audit


if __name__=='__main__':analyze()
