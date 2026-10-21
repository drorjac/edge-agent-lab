"""Validate report assets and JavaScript behavior in a JS engine with a minimal fake DOM.

This is a logic/syntax smoke check, not browser rendering or accessibility testing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.evidence = ""
        self.scripts = []
        self.assets = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "script":
            self.current = "evidence" if values.get("id") == "evidence" else "script"
            if self.current == "script":
                self.scripts.append("")
        if tag == "img":
            self.assets.append(values["src"])

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None

    def handle_data(self, data):
        if self.current == "evidence":
            self.evidence += data
        elif self.current == "script":
            self.scripts[-1] += data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/index.html"))
    args = parser.parse_args()
    parsed = ReportParser()
    parsed.feed(args.report.read_text())
    evidence = json.loads(parsed.evidence)
    for source in parsed.assets:
        path = args.report.parent / source
        if not path.is_file() or path.stat().st_size < 100:
            raise ValueError(f"Missing report asset: {source}")
    engine = shutil.which("node")
    if not engine:
        native = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")
        if not native.exists():
            raise RuntimeError("Install Node.js or use macOS JavaScriptCore for this smoke check")
        engine = str(native)
    prelude = """
if(typeof print==='undefined')globalThis.print=console.log;
const nodes={};
function node(id){if(!nodes[id])nodes[id]={innerHTML:'',textContent:'',value:'val_f1_mean',
  listeners:{},addEventListener(type,fn){this.listeners[type]=fn;}};return nodes[id]}
const tabs=['edge','steward','pilot','apple'].map(name=>({dataset:{tab:name},attrs:{},listeners:{},
  setAttribute(k,v){this.attrs[k]=v},addEventListener(type,fn){this.listeners[type]=fn}}));
const panels=['edge','steward','pilot','apple'].map(id=>({id,active:id==='edge',
  get classList(){return {toggle:(name,on)=>{this.active=on}}}}));
const document={getElementById:node,querySelectorAll(selector){
  if(selector==='[data-tab]')return tabs;if(selector==='.section')return panels;
  throw new Error('Unexpected selector: '+selector)}};
"""
    prelude += 'node("evidence").textContent=' + json.dumps(parsed.evidence) + ";\n"
    checks = """
function assert(ok,message){if(!ok)throw new Error(message)}
assert((node('models').innerHTML.match(/<tr>/g)||[]).length===data.aggregates.length,
       'model row count');
node('sort').value='bytes';node('sort').listeners.change();
const smallest=[...data.aggregates].sort((a,b)=>a.bytes-b.bytes)[0];
assert(node('models').innerHTML.startsWith('<tr><td>'+esc(smallest.name)+'</td>'),'size sorting');
tabs[1].listeners.click();
assert(panels.filter(p=>p.active).length===1&&panels[1].active,'tab panel switching');
assert(tabs[1].attrs['aria-selected']==='true'&&
       tabs[0].attrs['aria-selected']==='false','ARIA selection');
assert(esc('<script>&')==='&lt;script&gt;&amp;','HTML escaping');
assert(node('steward-results').innerHTML.includes('Actual example tool traces'),'trace display');
print(JSON.stringify({passed:true,candidates:data.aggregates.length,checks:6}));
"""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "report-smoke.js"
        path.write_text(prelude + "\n".join(parsed.scripts) + checks)
        result = subprocess.run([engine, str(path)], capture_output=True, text=True, check=True)
    print(result.stdout.strip())
    if evidence["edge"].get("in_progress"):
        raise ValueError("Report still contains an in-progress study snapshot")


if __name__ == "__main__":
    main()
