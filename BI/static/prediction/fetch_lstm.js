$(document).ready(() => {
    const req = () => {
        $.ajax({
            url: "/predictions-lstm/",
            type: "GET",
            success: (data) => {
                displayChart(data);
            },
            error: () => {
                alert(
                    "An error has been ocurred during load data from REST :(",
                );
            },
        });
    };

    function displayChart(data) {
        let dates = [];
        let t2m = [];
        let predictions = [];

        var mseValues = document.getElementById("MSEs");
        var msetrain_li = document.createElement("li");
        var msetest_li = document.createElement("li");
        msetrain_li.appendChild(
            document.createTextNode(`MSE for train data: ${data.mse_train}`),
        );
        msetest_li.appendChild(
            document.createTextNode(`MSE for test data: ${data.mse_test}`),
        );
        mseValues.appendChild(msetest_li);
        mseValues.appendChild(msetrain_li);

        data.data.forEach((item) => {
            dates.push(item.Date);
            t2m.push(item.t2m);
            predictions.push(item.Prediction);
        });

        new Chart(document.getElementById("graphic"), {
            type: "line",
            data: {
                labels: dates,
                datasets: [
                    {
                        data: t2m,
                        label: "Medium tempeature at two metters",
                        backgroundColor: "#FF6384",
                        borderColor: "#FF6384",
                    },
                    {
                        data: predictions,
                        label: "Prediction Model",
                        backgroundColor: "#9BD0F5",
                    },
                ],
            },
        });
    }
    req();
});
