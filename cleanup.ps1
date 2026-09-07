<#
    cleanup.ps1 - deletes the files and folders the local-only refactor removed.

    Run it ONCE from the project root:

        powershell -ExecutionPolicy Bypass -File .\cleanup.ps1

    Add -WhatIf to see what it would delete without deleting anything:

        powershell -ExecutionPolicy Bypass -File .\cleanup.ps1 -WhatIf

    Everything listed here is either dead code for a deployment target that no longer
    exists (Vercel / Pinecone / Cloudinary / Cohere / Jina / Vertex AI), a chunking
    strategy that was removed (fixed and hierarchical), a scaffold nothing imported, or
    generated state that MUST be rebuilt because the chunker changed.

    Nothing in src/ that the running app still needs is listed. If you are unsure, run it
    with -WhatIf first and read the list.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Remove-IfPresent {
    param([string]$RelativePath, [string]$Why)
    $full = Join-Path $root $RelativePath
    if (-not (Test-Path -LiteralPath $full)) {
        Write-Host ("  skip   {0,-52} (already gone)" -f $RelativePath) -ForegroundColor DarkGray
        return
    }
    if ($PSCmdlet.ShouldProcess($RelativePath, "Delete ($Why)")) {
        Remove-Item -LiteralPath $full -Recurse -Force
        Write-Host ("  DELETE {0,-52} {1}" -f $RelativePath, $Why) -ForegroundColor Yellow
    }
}

Write-Host "`n=== Cloud / serverless deployment (Vercel) ===" -ForegroundColor Cyan
Remove-IfPresent 'api'                              'Vercel entrypoint'
Remove-IfPresent 'vercel.json'                      'Vercel config'
Remove-IfPresent 'requirements-cloud.txt'           'cloud dependency set'
Remove-IfPresent 'SWITCHING.md'                     'local/cloud switching guide'

Write-Host "`n=== Remote embedding / re-rank / vector providers ===" -ForegroundColor Cyan
Remove-IfPresent 'src\ml\providers.py'              'Pinecone/Cohere/Jina/Gemini embeddings'
Remove-IfPresent 'src\ml\vertex.py'                 'Vertex AI LLM'
Remove-IfPresent 'src\services\vector_pinecone.py'  'Pinecone vector store'
Remove-IfPresent 'src\services\vector_chroma.py'    'merged into vectorstore.py'
Remove-IfPresent 'src\services\cloudinary_store.py' 'Cloudinary document store'
Remove-IfPresent 'src\services\cloud_documents.py'  'Cloudinary document registry'

Write-Host "`n=== Scripts for removed features ===" -ForegroundColor Cyan
Remove-IfPresent 'scripts\check_cloud.py'           'cloud deployment checker'
Remove-IfPresent 'scripts\diagnose_cloudinary.py'   'Cloudinary diagnostics'
Remove-IfPresent 'scripts\check_embeddings.py'      'Gemini/Pinecone embedding checker'
Remove-IfPresent 'scripts\ab_chunking.py'           'fixed-vs-semantic-vs-hierarchical A/B'
Remove-IfPresent 'scripts\draft_golden.py'          'one-off golden-question drafter'
Remove-IfPresent 'scripts\check_golden.py'          'one-off golden-question checker'
Remove-IfPresent 'scripts\make_test_pdf.py'         'fixture generator; tests make their own'

Write-Host "`n=== Tests for removed code ===" -ForegroundColor Cyan
Remove-IfPresent 'tests\test_gemini_embeddings_offline.py' 'Gemini embeddings removed'
Remove-IfPresent 'tests\test_vertex_llm_offline.py'        'Vertex AI removed'
Remove-IfPresent 'tests\test_parent_context_offline.py'    'hierarchical chunking removed'

Write-Host "`n=== Unwired scaffold ===" -ForegroundColor Cyan
Remove-IfPresent 'reference'                        'separate agent app; nothing imports it'

Write-Host "`n=== Docs superseded by README.md / CLAUDE.md ===" -ForegroundColor Cyan
Remove-IfPresent 'PLAN.md'                          'changelog for the removed hierarchical work'

Write-Host "`n=== Generated state (MUST be rebuilt: the chunker changed) ===" -ForegroundColor Cyan
Remove-IfPresent 'storage\chroma_db'                'index built with fixed chunking'
Remove-IfPresent 'data\users'                       'per-account uploads from old test accounts'
Remove-IfPresent 'eval\runs'                        'saved eval runs for removed strategies'
Remove-IfPresent 'eval\golden_draft.json'           'unreviewed draft questions'

Write-Host "`n=== __pycache__ (stale .pyc for deleted modules) ===" -ForegroundColor Cyan
Get-ChildItem -Path $root -Filter '__pycache__' -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\venv\\' } |
    ForEach-Object {
        $rel = $_.FullName.Substring($root.Length + 1)
        if ($PSCmdlet.ShouldProcess($rel, 'Delete (stale bytecode)')) {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force
            Write-Host ("  DELETE {0}" -f $rel) -ForegroundColor Yellow
        }
    }

Write-Host @"

------------------------------------------------------------------
Done.

The two textbook PDFs in data\ were LEFT IN PLACE. Delete them by hand
if you don't want them indexed.

Next steps:

  1. pip install -r requirements.txt
  2. Check .env against the new .env.example - RAG_MODE, VECTOR_STORE,
     EMBEDDINGS_PROVIDER, PINECONE_*, COHERE_*, JINA_*, CLOUDINARY_*,
     VERTEX_* and CHUNK_STRATEGY are all gone, and GROQ_MAX_TOKENS /
     GROQ_TEMPERATURE are now LLM_MAX_TOKENS / LLM_TEMPERATURE.
  3. python scripts\ingest.py --force     (rebuild with semantic chunking)
  4. python scripts\run.py
------------------------------------------------------------------
"@ -ForegroundColor Green
