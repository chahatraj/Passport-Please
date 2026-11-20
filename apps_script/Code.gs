// Passport Please: receives opted-in runs from the Colab notebook.
// Each run becomes one row in the "runs" sheet; its images go to a Drive folder
// and the row links to them. Deploy as a web app (see SETUP.md).

const FOLDER_NAME = 'Passport Please images';
const HEADER = ['time', 'session', 'run', 'part', 'modality', 'scene', 'target', 'detail', 'added',
                'prompt', 'question / instruction', 'labels', 'guessed origin', 'outputs',
                'image folder', 'image links'];

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const r = JSON.parse(e.postData.contents);
    const sheet = getSheet_();
    let folderUrl = '', links = [];
    if (r.images && r.images.length) {
      const folder = getRunFolder_(r.session, r.run);
      folderUrl = folder.getUrl();
      r.images.forEach(function (b64, i) {
        const blob = Utilities.newBlob(Utilities.base64Decode(b64), 'image/jpeg',
                                       r.part + '_' + r.modality + '_' + i + '.jpg');
        links.push(folder.createFile(blob).getUrl());
      });
    }
    const j = function (x) { return x === undefined || x === null ? '' : JSON.stringify(x); };
    sheet.appendRow([r.time, r.session, r.run, r.part, r.modality, r.scene || '', r.target || '',
                     r.detail || '', r.added || '', r.prompt || '', r.question || r.instruction || '',
                     j(r.labels), j(r.guessed), j(r.outputs).slice(0, 49000), folderUrl, links.join('\n')]);
    return ContentService.createTextOutput('ok');
  } finally {
    lock.releaseLock();
  }
}

function getSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName('runs');
  if (!sheet) {
    sheet = ss.insertSheet('runs');
    sheet.appendRow(HEADER);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function getRunFolder_(session, run) {
  const props = PropertiesService.getScriptProperties();
  let rootId = props.getProperty('ROOT_FOLDER_ID');
  let root;
  if (rootId) {
    root = DriveApp.getFolderById(rootId);
  } else {
    root = DriveApp.createFolder(FOLDER_NAME);
    props.setProperty('ROOT_FOLDER_ID', root.getId());
  }
  const it = root.getFoldersByName(session);
  const sessionFolder = it.hasNext() ? it.next() : root.createFolder(session);
  return sessionFolder.createFolder(run);
}

// Run this once from the editor to grant permissions and create the sheet and folder.
function setup() {
  getSheet_();
  getRunFolder_('setup-test', 'test');
}
