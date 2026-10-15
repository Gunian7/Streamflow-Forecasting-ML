from pathlib import Path
import re
import csv
import io

ROOT = Path(__file__).resolve().parent.parent
BASE = {
    'Date': ('日期', '观测日期，格式 YYYY-MM-DD。'),
    'Discharge': ('当天径流量', '当天测站流量，单位 cfs（立方英尺/秒）；作为原始观测保留，不参与新版特征选择。'),
    'Dayl': ('日照时长', '当天昼长，单位秒。'),
    'Prcp': ('降水量', '当天降水量，单位毫米/天。'),
    'Srad': ('短波辐射', '当天短波辐射通量，单位 W/m²。'),
    'Swe': ('积雪水当量', '积雪的等效水量，单位 kg/m²；当前样本中为零，不参与模型归一化。'),
    'Tmax': ('最高气温', '当天最高气温，单位摄氏度。'),
    'Tmin': ('最低气温', '当天最低气温，单位摄氏度。'),
    'Vp': ('水汽压', '当天水汽压，单位 Pa。'),
    'Year': ('年份', '日期所属公历年份。'),
    'Month': ('月份', '日期所属月份，取值 1 至 12。'),
    'Day': ('日', '日期在当月的日序号，取值 1 至 31。'),
    'DayOfWeek': ('星期编码', '星期一为 0，星期日为 6。'),
    'DayOfYear': ('年内日序', '日期在当年的日序号，1 起始，闰年可到 366。'),
    'Season_spring': ('春季独热编码', '3至5月取1，其余月份取0。四季四列每行恰有一列为1，无大小顺序。'),
    'Season_summer': ('夏季独热编码', '6至8月取1，其余月份取0。'),
    'Season_autumn': ('秋季独热编码', '9至11月取1，其余月份取0。'),
    'Season_winter': ('冬季独热编码', '12、1、2月取1，其余月份取0。'),
    'Season': ('四季编码', '冬季（12、1、2月）=0；春季（3至5月）=1；夏季（6至8月）=2；秋季（9至11月）=3。'),
    'Month_sin': ('月份正弦编码', 'sin(2π×月份/12)，表达月份周期，数值无单位。'),
    'Month_cos': ('月份余弦编码', 'cos(2π×月份/12)，与月份正弦编码配对表达月份周期。'),
    'DayOfYear_sin': ('年内日序正弦编码', 'sin(2π×年内日序/365.25)，使用现有程序的固定分母，闰年也采用此定义。'),
    'DayOfYear_cos': ('年内日序余弦编码', 'cos(2π×年内日序/365.25)，与年内日序正弦编码配对。'),
    'Feature': ('特征名称', '参与特征选择的候选变量名称；本表以中文显示。'),
    'Pearson_r': ('皮尔逊相关系数', '训练样本中该特征与未来第1天径流的线性相关系数，取值范围 [-1,1]。'),
    'Abs_Pearson_r': ('皮尔逊相关系数绝对值', '皮尔逊相关系数的绝对值，越大表示线性关联越强，不表示因果关系。'),
    'Mutual_Info': ('互信息得分', '使用 mutual_info_regression 估计该特征与未来第1天径流的统计依赖，能够反映非线性关联；不是百分比。'),
}
STATS = {'mean': '均值', 'std': '标准差', 'max': '最大值', 'min': '最小值'}


def describe(key):
    if key in BASE:
        return BASE[key]
    for suffix, label, formula in [
        ('_minmax', '最小最大归一化', '(x-训练集最小值)/(训练集最大值-训练集最小值)'),
        ('_zscore', 'Z分数标准化', '(x-训练集均值)/训练集标准差')]:
        if key.endswith(suffix):
            name, meaning = describe(key[:-len(suffix)])
            return name + '_' + label, meaning + ' 此列为无量纲变换值，公式：' + formula + '；复用训练样本拟合的 scaler。常量列由 sklearn 处理，不执行除零。'
    match = re.fullmatch(r'(.+)_lag(\d+)', key)
    if match:
        base, days = match.groups()
        name, meaning = describe(base)
        return name + '_滞后' + days + '天', '日期 t 对应 t-' + days + ' 天的观测值；单位与原变量相同。原变量说明：' + meaning
    match = re.fullmatch(r'(.+)_roll_(mean|std|max|min)_(\d+)', key)
    if match:
        base, stat, window = match.groups()
        name, _ = describe(base)
        explanation = '样本标准差，ddof=1' if stat == 'std' else STATS[stat]
        return name + '_过去' + window + '天' + STATS[stat], '先 shift(1)，再计算日期 t 前 ' + window + ' 天（t-' + window + ' 至 t-1）的' + explanation + '；不包含当天，单位与原变量相同。'
    raise ValueError('Unknown field: ' + key)


def translate(path):
    text = path.read_text(encoding='utf-8-sig')
    first, sep, rest = text.partition('\n')
    headers = next(csv.reader([first]))
    # Skip already translated files; metadata is recovered from the English snapshot.
    backup = path.with_name(path.stem + '_英文备份.csv')
    if backup.exists():
        original = backup.read_text(encoding='utf-8-sig')
        headers = next(csv.reader([original.splitlines()[0]]))
        text = original
        first, sep, rest = text.partition('\n')
    else:
        for col in headers:
            describe(col)
        backup.write_bytes(path.read_bytes())
    translated = [describe(col)[0] for col in headers]
    assert len(translated) == len(set(translated))
    stream = io.StringIO()
    csv.writer(stream, lineterminator='\n').writerow(translated)
    if 'Feature' in headers:
        rows = list(csv.reader(io.StringIO(text)))
        out = io.StringIO()
        writer = csv.writer(out, lineterminator='\n')
        writer.writerow(translated)
        for row in rows[1:]:
            row[headers.index('Feature')] = describe(row[headers.index('Feature')])[0]
            writer.writerow(row)
        result = out.getvalue()
    else:
        result = stream.getvalue() + rest
    path.write_text(result, encoding='utf-8-sig', newline='')
    before = list(csv.reader(io.StringIO(text)))
    after = list(csv.reader(io.StringIO(result)))
    assert len(before) == len(after)
    for old, new in zip(before[1:], after[1:]):
        if 'Feature' in headers:
            i = headers.index('Feature')
            assert old[:i] + old[i+1:] == new[:i] + new[i+1:]
        else:
            assert old == new
    return headers, len(after)-1


if __name__ == '__main__':
    targets = [ROOT/'results/processed_data/01047000_with_feature_values_独热编码版.csv', ROOT/'results/figures/fe_without_current_discharge/feature_selection_ranking.csv']
    sections = ['# CSV 字段说明', '', '## 使用说明', '', '原始模型输入 `01047000.csv` 未修改。本说明对应中文表头的特征数值表与不含当天径流的特征排名表。', '', '归一化参数来自有效训练样本（2000-01-08 至 2003-12-31）；输出仍保留全部原始日期。历史不足的滞后/窗口值及其归一化值留空，不填零；测试期的最小最大归一化结果可能超出 [0,1]，不截断。标准化不保证数据变为正态分布。', '', '原始当天径流及其归一化值在完整数值表中作为观测结果保留，但新版排名表不包含当天径流。特征选择目标是未来第1天径流，不是当天径流。', '', '每次重新运行特征工程脚本会输出英文表头；需要再次运行本地 `local_translate_csv.py` 生成中文展示版。英文备份只用于核对和恢复，数值不作修改。', '']
    for path in targets:
        headers, count = translate(path)
        sections += ['## ' + path.name, '', f'数据记录：{count} 行；字段：{len(headers)} 列。', '', '| 序号 | 中文字段 | 英文字段 | 含义与计算方式 |', '|---|---|---|---|']
        for i, col in enumerate(headers, 1):
            name, meaning = describe(col)
            sections.append(f'| {i} | {name} | `{col}` | {meaning} |')
        sections.append('')
        print(path.name, count, len(headers), 'all numeric strings unchanged')
    doc = ROOT/'CSV字段说明.md'
    doc.write_text('\n'.join(sections), encoding='utf-8')
    print(doc)
