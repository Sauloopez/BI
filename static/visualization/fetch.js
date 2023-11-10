$(document).ready(() => {
    var currentPage = 0;
    var pages;

    function fetchData() {
        $.ajax({
            url: '/nasa-data/',
            type: 'GET',
            success: (data) => {
                pages=data;
                displayData();
            },
            error: () => {
                alert('An error has been ocurred during load data from rest :(');
            }
        });
    }
    function displayData() {
        //These are the elements for the table for show the data
        var table = $('#table-results');
        var tableHead = $('<thead></thead>');
        var tableBody = $('<tbody class="table-group-divider"></tbody>');
        

        table.empty();

        var tableHeader = '<tr><th scope="col">Nro.</th><th scope="col">Date</th><th scope="col">Temperature medium at 2 meters</th><th scope="col">Temperature maximum at 2 meters</th><th scope="col">Temperature minimum at 2 meters</th><th scope="col">Range of temperature at 2 meters</th></tr>';
        table.append(tableHeader)
        pages[currentPage].forEach(function (item) {
            var row = '<tr>' +
                '<th scope="row">' + item.nro + '</th>' +
                '<td>' + item.date + '</td>' +
                '<td>' + item.t2m + ' °C</td>' +
                '<td>' + item.t2max + ' °C</td>' +
                '<td>' + item.t2min + ' °C</td>' +
                '<td>' + item.t2range + ' °C</td>' +
                '</tr>';
            tableBody.append(row);
        });

        table.append(tableHead);
        table.append(tableBody);
    }


    $('#prevPage').click(function () {
        if (currentPage > 0) {
            currentPage--;
            displayData();
        }
    });
    $('#nextPage').click(function () {
        if(currentPage < pages.length-1){
            currentPage++;
            displayData();
        }
    });
    
    fetchData(currentPage);
});



