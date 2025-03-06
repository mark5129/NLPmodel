document.addEventListener("DOMContentLoaded", function() {
    // List of files (this would typically be fetched dynamically, but GitHub Pages cannot serve directory listings)
    const files = [
        "1234567890_chart1.html", "1234567890_chart2.html",
        "0987654321_graph.html", "manualrun_report.html"
    ];

    let groups = {};
    files.forEach(file => {
        let match = file.match(/^\d{10}|manualrun/);
        if (match) {
            let key = match[0];
            if (!groups[key]) groups[key] = [];
            groups[key].push(file);
        }
    });

    let menu = document.getElementById("menu");
    Object.keys(groups).forEach(key => {
        let section = document.createElement("div");
        section.classList.add("menu-item");
        section.innerHTML = `<strong>${key}</strong>`;
        let list = document.createElement("ul");
        groups[key].forEach(file => {
            let item = document.createElement("li");
            item.innerHTML = `<a href="visualizations/${file}" target="_blank">${file}</a>`;
            list.appendChild(item);
        });
        section.appendChild(list);
        menu.appendChild(section);
    });
});