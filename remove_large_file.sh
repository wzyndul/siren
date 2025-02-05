#!/bin/bash

# File to remove from Git history
FILE_TO_REMOVE="experiment_scripts/logs/geo_experiment_1/summaries/events.out.tfevents.1725391731.bo129.1322358.0"

# Remove the file from Git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch $FILE_TO_REMOVE" \
  --prune-empty --tag-name-filter cat -- --all

# Remove the refs/original/
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Cleanup and optimize the repository
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push changes to remote
git push origin --force --all

echo "Large file removal process completed. Check the output for any errors."
