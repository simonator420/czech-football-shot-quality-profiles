# Release checklist

1. Confirm that `main` is clean and pushed.
2. Enable this repository in the Zenodo GitHub integration.
3. Change `Unreleased` in `CHANGELOG.md` to `v1.2.0` and add the release date.
4. Create the annotated semantic-version tag `v1.2.0`.
5. Push the tag and publish a GitHub release from it.
6. Confirm that Zenodo created the deposit and review its metadata before
   publishing it.
7. Add the resulting concept and version DOI badges and citations to the
   README and manuscript data-availability statement.
