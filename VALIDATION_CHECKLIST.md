# ML Factory Validation Checklist (Post-Critical Fixes)

- [ ] All models import without error
- [ ] PurgedKFold is used in all optimization files
- [ ] embargo_bars pulled from config (not hardcoded)
- [ ] Meta-labeling respects train/test split
- [ ] 4 optimization stages use different temporal windows
- [ ] Stacking unsafe mode removed
- [ ] DataFrame checksum uses full hash
