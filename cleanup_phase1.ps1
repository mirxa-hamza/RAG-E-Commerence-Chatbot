<#
    cleanup_phase1.ps1 - deletes the files and folders Phase 1 of PLAN.md retires.

    Run it ONCE from the project root, AFTER the edited/added files from this session have
    landed (config.py, main.py, auth.py, api/__init__.py, chunking.py, ml/llm.py,
    api/chat.py, the new services/text.py, .env, .env.example, requirements.txt):

        powershell -ExecutionPolicy Bypass -File .\cleanup_phase1.ps1

    Add -WhatIf to see what it would delete without deleting anything:

        powershell -ExecutionPolicy Bypass -File .\cleanup_phase1.ps1 -WhatIf

    Why each of these: the e-commerce pivot (PLAN.md) retires the whole PDF-upload,
    per-user-document feature - there is no more per-user upload, so there is nothing left
    for uploads.py/manifest.py/ownership.py/ingestion.py/documents.py to do, and pdf.py's
    only reusable pieces were split into the new services/text.py (already staged). The
    static/ frontend is replaced by a separate Next.js app in Phase 5, and storage/chroma_db
    holds vectors from the OLD PDF pipeline at the wrong embedding dimension for a fresh
    e-commerce index anyway.

    Nothing in src/ that the app still imports is listed here - this script was checked
    against the actual import graph (chunking.py, ml/llm.py and api/chat.py were updated to
    import from the new services/text.py instead of services/pdf.py; api/auth.py no longer
    imports services/ownership.py; api/__init__.py no longer registers documents.router)
    before being written, specifically so running it does not break the app's imports.
    If you are unsure, run it with -WhatIf first and read the list.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Remove-IfPresent {
    param([string]$RelativePath, [string]$Why)
    $full = Join-Path $root $RelativePath
    if (-not (Test-Path -LiteralPath $full)) {
        Write-Host ("  skip   {0,-46} (already gone)" -f $RelativePath) -ForegroundColor DarkGray
        return
    }
    if ($PSCmdlet.ShouldProcess($RelativePath, "Delete ($Why)")) {
        Remove-Item -LiteralPath $full -Recurse -Force
        Write-Host ("  DELETE {0,-46} {1}" -f $RelativePath, $Why) -ForegroundColor Yellow
    }
}

Write-Host "`n=== PDF-upload / per-user-document feature (retired - shared catalog now) ===" -ForegroundColor Cyan
Remove-IfPresent 'src\services\pdf.py'         'PDF extraction; reusable bits moved to services\text.py'
Remove-IfPresent 'src\services\uploads.py'     'per-user upload boundary; no more per-user uploads'
Remove-IfPresent 'src\services\manifest.py'    'PDF ingest manifest; Phase 2 tracks products/reviews instead'
Remove-IfPresent 'src\services\ownership.py'   'per-user document ownership; the catalog is shared, not owned'
Remove-IfPresent 'src\services\ingestion.py'   'PDF background-ingest job; replaced by scripts\ingest_ecommerce.py in Phase 2'
Remove-IfPresent 'src\api\documents.py'        'PDF upload/list/delete routes'
Remove-IfPresent 'scripts\ingest.py'           'PDF ingest CLI; replaced by scripts\ingest_ecommerce.py in Phase 2'
Remove-IfPresent 'scripts\verify_index.py'     'PDF index verifier; used ingestion.py + manifest.py, both gone'

Write-Host "`n=== Frontend (replaced by a separate Next.js app in Phase 5) ===" -ForegroundColor Cyan
Remove-IfPresent 'src\static'                  'vanilla HTML/JS/CSS frontend; PLAN.md Phase 5 is Next.js + Tailwind'

Write-Host "`n=== Docs / eval superseded by the e-commerce pivot ===" -ForegroundColor Cyan
Remove-IfPresent 'report'                      'leftover PDF-chunking-strategy report; already dead weight before Phase 1'
Remove-IfPresent 'eval'                        'PDF hit-rate@k harness; does not transfer to product/review search'

Write-Host "`n=== Generated state (MUST be rebuilt: wrong embedding dimension / wrong content) ===" -ForegroundColor Cyan
Remove-IfPresent 'storage\chroma_db'           'PDF-era index (rag_gemini_768, 768-dim); Phase 2 rebuilds amazon_fashion_reviews_384 at 384-dim'

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
Done. Phase 1 (PLAN.md) is complete once this has run.

Next steps:

  1. pip install -r requirements.txt        (drops pymupdf/python-multipart, adds langdetect)
  2. Make sure MongoDB is running locally (mongodb://localhost:27017) - .env now points
     there instead of the old Atlas cluster.
  3. python scripts\run.py                  (the app should now start cleanly; sign-in
     works once MongoDB is reachable, but there is no catalog to chat about yet - that's
     Phase 2)
  4. When you're ready, say so and Phase 2 (the Amazon Fashion ingestion pipeline) starts.
------------------------------------------------------------------
"@ -ForegroundColor Green
