c:\python311\python -m venv .venv
.\.venv\scripts\activate.ps1
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip list --format=freeze > requirements-frozen.txt
deactivate

# Remove the lines that start with 'tailwind' from the requirements-frozen.txt file
# The pip install in the dockerfile will try to download the tailwind package from PyPI, and it doesn't exist there.

$content = Get-Content "requirements-frozen.txt" -encoding UTF8

# Use -replace with a regular expression to remove lines starting with 'tailwind'
# The regular expression '^tailwind.*$' will match any line that starts with 'tailwind'
$updatedContent = $content -replace '^tailwind.*$', ''

# Write the updated content back to the file
$updatedContent | Set-Content  "requirements-frozen.txt" -Encoding UTF8
