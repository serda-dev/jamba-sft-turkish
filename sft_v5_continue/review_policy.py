"""Development-only selection rules; frozen external tests never select weights."""
def eligible(baseline_dev,dev,baseline_math,math_report,baseline_loss,loss,baseline_conversation,conversation,cfg):
 p=cfg['pilot_policy'];complete=dev['n']==128 and math_report['n']==143 and loss['complete'] and conversation['n']==baseline_conversation['n']==15
 other_loss=loss['complete'] and set(loss['scores'])==set(baseline_loss['scores']) and all(
  loss['scores'][k]['tokens']==v['tokens'] and loss['scores'][k]['mean_loss']<=max(v['mean_loss']*p['family_loss_max_ratio'],v['mean_loss']+p['family_loss_absolute_allowance'])
  for k,v in baseline_loss['scores'].items() if k!='conversation')
 return {'complete':complete,'math_retained':math_report['correct']>=baseline_math['correct']-p['natural_math_max_drop'],'stopping':dev['truncated']<=baseline_dev['truncated']+p['dev_max_extra_truncation'],'other_family_losses_retained':other_loss,'conversation_generated_retained':conversation['correct']>=baseline_conversation['correct'] and conversation['truncated']<=baseline_conversation['truncated']}

def pause_ready(report):
 return isinstance(report,dict) and report.get('tag')=='dev_1997' and report.get('n')==128
