# Collecting runs in a Google Sheet

About 5 minutes, once, in the Google account that should own the data.

1. Create a new Google Sheet (for example "Passport Please runs").
2. In the sheet: **Extensions → Apps Script**. Delete the starter code and paste `Code.gs`. Save.
3. In the function menu at the top, pick **setup** and press **Run**. Approve the permissions
   (Sheets and Drive). This creates the `runs` tab and a Drive folder "Passport Please images".
   You can delete the `setup-test` folder inside it afterwards.
4. **Deploy → New deployment → Web app.** Execute as: **Me**. Who has access: **Anyone**. Deploy and
   copy the web app URL (it ends in `/exec`).
5. Paste the URL into `LOG_URL` at the top of `build_activity.py`, run `python build_activity.py`,
   commit and push.

What arrives: one row per run (session id, part, modality,
scene, detail or added words, prompt, labels, generated text), plus that run's images as 320 px JPEGs
in `Passport Please images/<session>/<run>/`, linked from the row.

The same web app also hands out the Part 1 country to each student in rotation (`doGet`), so
countries are balanced. To restart the rotation, delete the `ASSIGN_COUNT` script property
(Project Settings → Script Properties).

If you change `Code.gs` later: **Deploy → Manage deployments → Edit → Version: New version**, so the
URL stays the same.
