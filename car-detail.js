(function () {
    var fallbackImage = "images/run-car/car.png";
    var specificationLabels = [
        ["displacement", "Displacement"],
        ["cylinders", "Cylinders"],
        ["fuel_tank_capacity", "Fuel tank capacity"],
        ["height", "Height"],
        ["length", "Length"],
        ["width", "Width"],
        ["seating_capacity", "Seating capacity"],
        ["power", "Power"],
        ["torque", "Torque"]
    ];

    function showError(message) {
        document.getElementById("detail-loading").hidden = true;
        var error = document.getElementById("detail-error");
        error.textContent = message;
        error.hidden = false;
    }

    function formatPrice(value) {
        return new Intl.NumberFormat("en-IN", {
            style: "currency",
            currency: "INR",
            maximumFractionDigits: 0
        }).format(Number(value));
    }

    async function loadDetail() {
        var parameters = new URLSearchParams(window.location.search);
        var carId = parameters.get("car_id");
        if (!carId || !/^\d+$/.test(carId)) {
            showError("A valid car ID is required. Return to the catalog and select a car.");
            return;
        }

        try {
            var car = await CarWorldAPI.getCar(carId);
            document.title = car.manufacturer + " " + car.model + " | CarWorld";
            document.getElementById("detail-title").textContent = car.manufacturer + " " + car.model;
            document.getElementById("detail-variant").textContent = car.variant;
            document.getElementById("detail-price").textContent = formatPrice(car.ex_showroom_price);
            document.getElementById("detail-fuel").textContent = car.fuel_type;
            document.getElementById("detail-body").textContent = car.body_type;
            document.getElementById("detail-transmission").textContent = car.transmission_type;

            var imagePath = parameters.get("image") || fallbackImage;
            if (!imagePath.startsWith("images/") || imagePath.includes("..")) imagePath = fallbackImage;
            var image = document.getElementById("detail-image");
            image.src = imagePath;
            image.alt = car.manufacturer + " " + car.model;
            image.onerror = function () { image.onerror = null; image.src = fallbackImage; };

            var specifications = document.getElementById("detail-specifications");
            specificationLabels.forEach(function (item) {
                var column = document.createElement("div");
                column.className = "col-md-4 col-6 border p-2 bg-light";
                var label = document.createElement("p");
                label.className = "m-0 font-weight-bold text-uppercase";
                label.textContent = item[1];
                var value = document.createElement("p");
                value.className = "m-0";
                value.textContent = car[item[0]];
                column.append(label, value);
                specifications.append(column);
            });

            document.getElementById("detail-loading").hidden = true;
            document.getElementById("detail-content").hidden = false;
        } catch (error) {
            if (error.status === 404) {
                showError("We couldn't find this car in the catalog. Return to the catalog and choose another listing.");
            } else {
                var message = error && error.message && /CarWorld API URL is not configured/i.test(error.message)
                    ? error.message + " Run start-frontend.cmd from the project folder, then reload this page."
                    : "We couldn't load this car right now. Please try again shortly.";
                showError(message);
            }
        }
    }

    document.addEventListener("DOMContentLoaded", loadDetail);
})();
