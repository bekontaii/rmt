"""Scientific final charts: ReportLab VerticalBarChart + exact CI overlays.
No models. SVG is rasterized with the existing bundled Node Sharp installation.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import reportlab
from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors

METHODS = ['fixed', 'overlap', 'semantic']
LABELS = ['Fixed', 'Overlap', 'Semantic']
W, H = 576, 360  # 8 x 5 inches, 1.6 ratio; insert at 6 x 3.75 inches in Word.
DPI, PIXELS = 220, (1760, 1100)
BUNDLE = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def ordered(rows, metrics):
    lookup = {row['strategy']: row for row in rows}
    require(len(rows) == len(lookup) == 3 and set(lookup) == set(METHODS), 'Expected exactly the three method rows')
    result = [lookup[s] for s in METHODS]
    require(len({row['n_questions'] for row in result}) == 1, 'Method denominators differ')
    for row in result:
        for key in metrics:
            v, lo, hi = [float(row[k]) for k in [key, key+'_ci_low', key+'_ci_high']]
            require(all(math.isfinite(x) for x in [v,lo,hi]) and 0 <= v <= 1 and 0 <= lo <= hi <= 1,
                    'Invalid point estimate or interval: '+key)
    return result


def drawing_for(rows, metrics, panel_titles):
    drawing = Drawing(W,H)
    drawing.add(Rect(0,0,W,H,fillColor=colors.white,strokeColor=None))
    count = len(metrics)
    left, right, gap = 42, 16, 35 if count == 3 else 45
    width = (W-left-right-gap*(count-1))/count
    trace = []
    for panel, (key,title) in enumerate(zip(metrics,panel_titles)):
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = left+panel*(width+gap), 58, width, 242
        chart.data = [tuple(float(row[key])*100 for row in rows)]
        chart.categoryAxis.categoryNames = LABELS
        chart.categoryAxis.style = 'parallel'
        chart.categoryAxis.labels.fontName = 'Helvetica'
        chart.categoryAxis.labels.fontSize = 11.5
        chart.categoryAxis.labels.dy = -13
        chart.categoryAxis.tickDown = 0
        chart.categoryAxis.strokeWidth = .6
        chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0,100,20
        chart.valueAxis.labelTextFormat = '%d%%'
        chart.valueAxis.labels.fontName = 'Helvetica'
        chart.valueAxis.labels.fontSize = 10.5
        chart.valueAxis.visibleGrid = 1
        chart.valueAxis.gridStrokeColor = colors.Color(.86,.86,.86)
        chart.valueAxis.gridStrokeWidth = .4
        chart.valueAxis.strokeWidth = .6
        chart.valueAxis.tickLeft = 3
        chart.valueAxis.tickRight = 0
        chart.barWidth, chart.groupSpacing, chart.barSpacing = 25,20,0
        for column,shade in enumerate([.80,.60,.35]):
            chart.bars[(0,column)].fillColor = colors.Color(shade,shade,shade)
            chart.bars[(0,column)].strokeColor = colors.black
            chart.bars[(0,column)].strokeWidth = .5
        # Use the standard chart's own bar layout and axis transform for overlays.
        drawing.add(chart.draw())
        drawing.add(String(chart.x+width/2,333,f'({chr(97+panel)}) {title}',
                           fontName='Helvetica-Bold',fontSize=14,textAnchor='middle',fillColor=colors.black))
        for column,row in enumerate(rows):
            x,y,bw,bh = chart._barPositions[0][column]
            center = x+bw/2
            value = float(row[key])*100
            lo,hi = float(row[key+'_ci_low'])*100,float(row[key+'_ci_high'])*100
            low_y,high_y = chart.valueAxis.scale(lo),chart.valueAxis.scale(hi)
            require(abs((y+bh)-chart.valueAxis.scale(value)) < 1e-6, 'Bar coordinate/axis transform mismatch')
            drawing.add(Line(center,low_y,center,high_y,strokeColor=colors.black,strokeWidth=1.0))
            for yy in [low_y,high_y]:
                drawing.add(Line(center-5,yy,center+5,yy,strokeColor=colors.black,strokeWidth=1.0))
            label_y = max(high_y,y+bh)+8
            drawing.add(String(center,label_y,f'{value:.2f}',fontName='Helvetica',fontSize=11.5,
                               textAnchor='middle',fillColor=colors.black))
            trace.append(dict(metric=key,strategy=row['strategy'],estimate=value,ci_low=lo,ci_high=hi,
                              bar_center_x=center,bar_top_y=y+bh,ci_low_y=low_y,ci_high_y=high_y))
    return drawing,trace


def rasterize(svg,png,node,packages):
    script = """const sharp=require(process.env.SHARP_MODULE);
const [src,dst]=process.argv.slice(1);
sharp(src,{density:220}).resize({width:1760,height:1100,fit:'fill'})
.flatten({background:'#ffffff'}).withMetadata({density:220}).png()
.toFile(dst).then(info=>process.stdout.write(JSON.stringify(info)))
.catch(e=>{process.stderr.write(String(e));process.exit(1)});"""
    env=dict(os.environ,SHARP_MODULE=str(packages/'sharp'))
    result=subprocess.run([str(node),'-e',script,str(svg),str(png)],env=env,check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--retrieval-only',action='store_true',help='Render only actual retrieval data while generation is pending')
    parser.add_argument('--node',type=Path,default=BUNDLE/'node/bin/node')
    parser.add_argument('--node-modules',type=Path,default=BUNDLE/'node/node_modules')
    args=parser.parse_args();out=args.out.resolve()
    require(Path(__file__).resolve().parent in out.parents,'Write plots beneath report_tools only')
    analysis=json.loads(args.analysis.read_text(encoding='utf-8'))
    tables=analysis['tables']
    retrieval=tables.get('retrieval',tables.get('all_retrieval'))
    require(retrieval is not None,'No actual retrieval table found')
    retrieval=ordered(retrieval,['hit_at_1','hit_at_3','hit_at_5'])
    pending=[('figure1_retrieval',retrieval,['hit_at_1','hit_at_3','hit_at_5'],['Hit@1','Hit@3','Hit@5'])]
    if not args.retrieval_only:
        require('generation' in tables,'Actual completed generation table is required; no fabricated data are substituted')
        generation=ordered(tables['generation'],['exact_match','f1'])
        pending.append(('figure2_generation',generation,['exact_match','f1'],['Exact match','Token F1']))
    out.mkdir(parents=True,exist_ok=True)
    ledger=dict(source_analysis=str(args.analysis.resolve()),source_sha256=sha(args.analysis),
                plot_script_sha256=sha(__file__),reportlab_version=reportlab.Version,
                canvas_inches=[8,5],png_pixels=list(PIXELS),png_dpi=DPI,word_insert_inches=[6,3.75],
                renderer='ReportLab VerticalBarChart + ReportLab renderSVG + bundled Node Sharp',figures={})
    clusters=analysis.get('article_clusters',20)
    if analysis.get('mode')=='development':clusters=3
    for stem,rows,metrics,titles in pending:
        drawing,trace=drawing_for(rows,metrics,titles)
        svg,png=out/(stem+'.svg'),out/(stem+'.png')
        renderSVG.drawToFile(drawing,str(svg))
        info=rasterize(svg,png,args.node,args.node_modules)
        require(info['width']==PIXELS[0] and info['height']==PIXELS[1],'Unexpected raster dimensions')
        if stem.startswith('figure1'):
            caption=f'Figure 1. Answer-span Hit@1, Hit@3 and Hit@5 on {rows[0]["n_questions"]:,} paired retrieval questions per method.'
        else:
            caption=f'Figure 2. Exact match and mean token F1 on {rows[0]["n_questions"]:,} paired generation questions per method.'
        caption+=f' Bars show observed percentages; whiskers are marginal 95% percentile intervals from 2,000 whole-article bootstrap draws ({clusters} article clusters; seed 42), conditional on the fixed corpus and models.'
        ledger['figures'][stem]=dict(png=str(png),png_sha256=sha(png),svg=str(svg),svg_sha256=sha(svg),
                                    caption=caption,coordinate_trace=trace,raster_info=info)
    write(out/'plot_manifest.json',ledger)
    print(json.dumps(dict(outputs=str(out),figures=list(ledger['figures']),renderer=ledger['renderer']),indent=2))


if __name__=='__main__':main()
