import { pathToFileURL } from 'node:url';
const skill='C:/Users/ndvuo/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const result=await finalizePresentation({
 workspaceDir:'D:/SBIR/RSGG',
 candidatePath:'D:/SBIR/RSGG/.codex-finalizer/context_20261024/candidate.pptx',
 finalPath:'D:/SBIR/RSGG/outputs/context_scientific_slides_20261024/Context_Method_Results_SG_20261024.pptx',
 pythonExecutable:'C:/Users/ndvuo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 explicitTotalSlideCount:4,
 requiredNativeTableOwnerSlides:[1],
 layoutArgs:['--expected-slide-size-emu','15240000,8572500','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','1'],
 fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
 receiptPath:'D:/SBIR/RSGG/.codex-finalizer/context_20261024/validation.json'
});
console.log(JSON.stringify(result,null,2));
