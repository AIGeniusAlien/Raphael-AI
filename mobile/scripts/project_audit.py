"""Linux/macOS structural audit;passing is not an iOS compile,simulator or signing result."""
import argparse
import json
from pathlib import Path
import plistlib
import re
import struct
import xml.etree.ElementTree as ET

TOKEN=re.compile(r'\s+|/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|[{}()=;,]|[^\s{}()=;,"]+',re.S)
OBJECT_ID=re.compile(r'[0-9A-F]{24}')


def parse_project(text):
    tokens=[];position=0
    while position<len(text):
        match=TOKEN.match(text,position)
        if not match:raise ValueError('Invalid OpenStep token.')
        value=match.group();position=match.end()
        if value.isspace() or value.startswith(('//','/*')):continue
        tokens.append(value)
    position=0
    def take(expected=None):
        nonlocal position
        if position>=len(tokens):raise ValueError('Unexpected project end.')
        value=tokens[position];position+=1
        if expected is not None and value!=expected:raise ValueError('Unexpected project punctuation.')
        return value
    def value():
        token=take()
        if token=='{':
            result={}
            while tokens[position]!='}':
                key=take();key=json.loads(key) if key.startswith('"') else key
                if key in result:raise ValueError('Duplicate project key.')
                take('=');result[key]=value();take(';')
            take('}');return result
        if token=='(':
            result=[]
            while tokens[position]!=')':result.append(value());take(',')
            take(')');return result
        return json.loads(token) if token.startswith('"') else int(token) if token.isdigit() else token
    result=value()
    if position!=len(tokens):raise ValueError('Unexpected trailing project tokens.')
    return result


def audit(ios_root):
    ios_root=Path(ios_root);project=parse_project((ios_root/'Raphael.xcodeproj/project.pbxproj').read_text());objects=project['objects']
    if objects[project['rootObject']]['isa']!='PBXProject':raise ValueError('Missing project root.')
    def references(value):
        if isinstance(value,str) and OBJECT_ID.fullmatch(value) and value not in objects:raise ValueError('Dangling project object reference.')
        if isinstance(value,dict):
            for nested in value.values():references(nested)
        if isinstance(value,list):
            for nested in value:references(nested)
    references(project)
    refs={}
    def visit(group_id,base):
        group=objects[group_id];base=base/group.get('path','')
        for child in group.get('children',[]):
            obj=objects[child]
            if obj['isa']=='PBXGroup':visit(child,base)
            elif obj['isa']=='PBXFileReference' and obj.get('sourceTree')=='<group>':
                path=base/obj['path']
                if not path.exists():raise ValueError('Referenced source/resource/configuration file is absent.')
                refs[child]=path
    visit(objects[project['rootObject']]['mainGroup'],ios_root)
    targets=[obj for obj in objects.values() if obj['isa']=='PBXNativeTarget']
    expected={'Raphael':'com.apple.product-type.application','RaphaelTests':'com.apple.product-type.bundle.unit-test','RaphaelUITests':'com.apple.product-type.bundle.ui-testing'}
    if {target['name']:target['productType'] for target in targets}!=expected:raise ValueError('Required app and native test targets missing.')
    app=next(target for target in targets if target['name']=='Raphael')
    for target in targets:
        configs=objects[target['buildConfigurationList']]['buildConfigurations']
        if {objects[key]['name'] for key in configs}!={'Debug','Release'}:raise ValueError('Target build configurations missing.')
        sources=[]
        for phase_id in target['buildPhases']:
            phase=objects[phase_id]
            if phase['isa']=='PBXSourcesBuildPhase':sources.extend(refs[objects[file]['fileRef']] for file in phase['files'])
        folder=ios_root/target['name']
        if set(sources)!=set(folder.glob('*.swift')):raise ValueError('Native target does not build every expected Swift source.')
    resources=[]
    for phase_id in app['buildPhases']:
        phase=objects[phase_id]
        if phase['isa']=='PBXResourcesBuildPhase':resources.extend(refs[objects[file]['fileRef']].name for file in phase['files'])
    if not {'PrivacyInfo.xcprivacy','Assets.xcassets'}<=set(resources):raise ValueError('Privacy/assets not bundled.')
    if not any(objects[phase]['isa']=='PBXShellScriptBuildPhase' and 'validate_release.py' in objects[phase]['shellScript'] for phase in app['buildPhases']):raise ValueError('Release configuration gate missing.')
    if not (ios_root.parent/'scripts/validate_release.py').exists():raise ValueError('Release validation script absent.')
    scheme=ET.parse(ios_root/'Raphael.xcodeproj/xcshareddata/xcschemes/Raphael.xcscheme')
    listed={reference.attrib['BlueprintIdentifier'] for reference in scheme.findall('.//BuildableReference')}
    target_ids={key for key,obj in objects.items() if obj['isa']=='PBXNativeTarget'}
    if listed!=target_ids:raise ValueError('Shared scheme does not cover all targets.')
    if len(scheme.findall('.//TestableReference'))!=2:raise ValueError('Native tests not selected by scheme.')
    info=plistlib.loads((ios_root/'Raphael/Info.plist').read_bytes())
    if info['NSAppTransportSecurity'].get('NSAllowsArbitraryLoads') is not False:raise ValueError('Insecure transport exception.')
    if info['WKAppBoundDomains']!=['$(RAPHAEL_SERVER_HOST)']:raise ValueError('App bound origin configuration missing.')
    if not isinstance(info['ITSAppUsesNonExemptEncryption'],bool):raise ValueError('Encryption declaration must use the required Boolean plist type.')
    privacy=plistlib.loads((ios_root/'Raphael/PrivacyInfo.xcprivacy').read_bytes())
    if privacy.get('NSPrivacyTracking') is not False:raise ValueError('Unexpected tracking declaration.')
    if not any(row['NSPrivacyCollectedDataType']=='NSPrivacyCollectedDataTypeHealth' for row in privacy['NSPrivacyCollectedDataTypes']):raise ValueError('Health-data collection inventory omitted.')
    icons=ios_root/'Raphael/Assets.xcassets/AppIcon.appiconset';catalog=json.loads((icons/'Contents.json').read_text())
    marketing=False
    for entry in catalog['images']:
        data=(icons/entry['filename']).read_bytes()
        if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('App icon is not a PNG.')
        size=struct.unpack('>II',data[16:24]);pixels=round(float(entry['size'].split('x')[0])*int(entry['scale'][0]))
        if size!=(pixels,pixels) or data[25]!=2:raise ValueError('App icon dimensions/opaque RGB format invalid.')
        if entry['idiom']=='ios-marketing' and size==(1024,1024):marketing=True
    if not marketing:raise ValueError('Marketing app icon absent.')
    return {'structural_audit':'passed','project_objects':len(objects),'native_targets':len(targets),'source_files':len(list((ios_root/'Raphael').glob('*.swift'))),'app_icon_entries':len(catalog['images']),'xcode_compile':'NOT_RUN','simulator_device_signing':'NOT_RUN'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--ios-root',type=Path,default=Path(__file__).resolve().parents[1]/'ios');args=parser.parse_args()
    try:result=audit(args.ios_root)
    except (ValueError,KeyError,IndexError,OSError) as error:print(json.dumps({'structural_audit':'failed','error_type':type(error).__name__}));return 1
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
