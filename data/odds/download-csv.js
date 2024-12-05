// Use on https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-20xx-yy/

(function() {
  // Select the table you want to extract. Adjust the selector as needed.
  const table = document.querySelector('table');
  if (!table) {
      console.error('No table found on the page.');
      return;
  }

  // Extract rows and cells
  const rows = Array.from(table.rows);
  const csvData = rows.map(row => {
      const cells = Array.from(row.cells);
      return cells.map(cell => `"${cell.textContent.trim()}"`).join(',');
  }).join('\n');

  // Create a Blob with the CSV data
  const blob = new Blob([csvData], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);

  // Create a download link
  const a = document.createElement('a');
  a.href = url;
  a.download = 'table_data.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  console.log('CSV file has been generated and downloaded.');
})();
