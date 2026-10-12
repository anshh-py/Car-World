(function () {
    var fallbackImage = "images/run-car/car.png";
    var fuelTypes = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric", "CNG + Petrol"];
    var bodyTypes = ["SUV", "Sedan", "Hatchback", "Coupe", "MPV", "MUV", "Convertible", "Crossover", "Not Mentioned", "Crossover, SUV", "Sports", "Pick-up", "Convertible, Sports", "Coupe, Sedan", "Hatchback, Sports", "Crossover, Sedan", "Convertible, Coupe"];
    var transmissionTypes = ["Manual", "Automatic", "AMT", "DCT", "CVT", "Not Mentioned"];

    function normalizeName(value) {
        return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    }

    function collectImages(root) {
        var images = [];
        root.querySelectorAll(".card").forEach(function (card) {
            var title = card.querySelector(".card-title") || card.querySelector("h4");
            var image = card.querySelector("img");
            if (title && image) images.push({ name: normalizeName(title.textContent), src: image.getAttribute("src") });
        });
        root.querySelectorAll('[id="newcarline"]').forEach(function (row) {
            var title = row.querySelector("h4");
            var image = row.querySelector("img");
            if (title && image) images.push({ name: normalizeName(title.textContent), src: image.getAttribute("src") });
        });
        return images;
    }

    function imageForCar(car, images) {
        var name = normalizeName(car.manufacturer + " " + car.model);
        var match = images.find(function (item) {
            return item.name === name || name.indexOf(item.name) >= 0 || item.name.indexOf(name) >= 0;
        });
        return match ? match.src : fallbackImage;
    }

    function formatPrice(value) {
        return new Intl.NumberFormat("en-IN", {
            style: "currency",
            currency: "INR",
            maximumFractionDigits: 0
        }).format(Number(value));
    }

    function makeCarLink(car, image) {
        return "CarDetails.html?car_id=" + encodeURIComponent(car.id) + "&image=" + encodeURIComponent(image);
    }

    function renderCards(container, cars, images) {
        container.replaceChildren();
        cars.forEach(function (car) {
            var image = imageForCar(car, images);
            var column = document.createElement("div");
            column.className = "col-lg-4 col-sm-6 mb-3";

            var link = document.createElement("a");
            link.className = "carworld-card-link";
            link.href = makeCarLink(car, image);

            var card = document.createElement("div");
            card.className = "card h-100";
            var photo = document.createElement("img");
            photo.className = "card-img-top carworld-card-image";
            photo.src = image;
            photo.alt = car.manufacturer + " " + car.model;
            photo.onerror = function () { photo.onerror = null; photo.src = fallbackImage; };

            var body = document.createElement("div");
            body.className = "card-body bg-light text-muted";
            var title = document.createElement("h5");
            title.className = "card-title";
            title.textContent = car.manufacturer + " " + car.model + " " + car.variant;
            var price = document.createElement("b");
            price.textContent = formatPrice(car.ex_showroom_price);
            var details = document.createElement("p");
            details.className = "mb-0 mt-2";
            details.textContent = [car.fuel_type, car.body_type, car.transmission_type].join(" · ");

            body.append(title, price, details);
            card.append(photo, body);
            link.append(card);
            column.append(link);
            container.append(column);
        });
    }

    function toggleState(loading, error, isLoading, message) {
        loading.hidden = !isLoading;
        error.hidden = !message;
        if (message) error.textContent = message;
    }

    function setupPager(ids, onPage) {
        var previous = document.getElementById(ids.previous);
        var next = document.getElementById(ids.next);
        var label = document.getElementById(ids.label);
        var state = { page: 1, totalPages: 0, previous: previous, next: next, label: label };
        previous.addEventListener("click", function () { if (state.page > 1) onPage(state.page - 1); });
        next.addEventListener("click", function () { if (state.page < state.totalPages) onPage(state.page + 1); });
        return function (page, pageSize, totalItems, totalPages) {
            state.page = page;
            state.totalPages = totalPages;
            previous.disabled = page <= 1 || totalPages === 0;
            next.disabled = page >= totalPages || totalPages === 0;
            label.textContent = totalItems ? "Page " + page + " of " + totalPages + " · " + totalItems + " cars" : "No cars found";
        };
    }

    function addOptions(select, values, firstLabel) {
        select.replaceChildren(new Option(firstLabel, ""));
        values.forEach(function (value) { select.add(new Option(value, value)); });
    }

    async function setupHomeCatalog() {
        var catalog = document.getElementById("car-api-catalog");
        if (!catalog) return;

        var loading = document.getElementById("car-api-loading");
        var error = document.getElementById("car-api-error");
        var results = document.getElementById("car-api-results");
        var filterForm = document.getElementById("car-filter-form");
        var searchForm = document.getElementById("car-search-form");
        var searchInput = document.getElementById("car-search-query");
        var manufacturerSelect = document.getElementById("filter-manufacturer");
        var modelSelect = document.getElementById("filter-model");
        var images = collectImages(document.getElementById("popular"));
        var page = 1;
        var searchQuery = new URLSearchParams(window.location.search).get("q") || "";
        searchInput.value = searchQuery;

        addOptions(document.getElementById("filter-fuel"), fuelTypes, "All fuel types");
        addOptions(document.getElementById("filter-body"), bodyTypes, "All body types");
        addOptions(document.getElementById("filter-transmission"), transmissionTypes, "All transmissions");

        try {
            var manufacturers = await CarWorldAPI.getManufacturers();
            manufacturers.forEach(function (manufacturer) {
                manufacturerSelect.add(new Option(manufacturer.name, manufacturer.name));
            });
        } catch (requestError) {
            toggleState(loading, error, false, "Could not load manufacturers: " + requestError.message);
        }

        manufacturerSelect.addEventListener("change", async function () {
            modelSelect.disabled = !manufacturerSelect.value;
            addOptions(modelSelect, [], manufacturerSelect.value ? "All models" : "Choose a manufacturer first");
            if (!manufacturerSelect.value) return;
            try {
                var response = await CarWorldAPI.getManufacturerModels(manufacturerSelect.value);
                response.items.forEach(function (model) { modelSelect.add(new Option(model.name, model.name)); });
            } catch (requestError) {
                toggleState(loading, error, false, "Could not load models: " + requestError.message);
            }
        });

        var updatePager = setupPager({ previous: "car-page-previous", next: "car-page-next", label: "car-page-label" }, loadPage);

        function filterParameters() {
            var parameters = {};
            new FormData(filterForm).forEach(function (value, key) {
                if (value !== "" && key !== "sort") parameters[key] = value;
            });
            return parameters;
        }

        async function loadPage(nextPage) {
            page = nextPage;
            toggleState(loading, error, true, "");
            results.replaceChildren();
            try {
                var response = searchQuery
                    ? await CarWorldAPI.searchCars({ q: searchQuery, page: page, page_size: 20, sort: "name" })
                    : await CarWorldAPI.getCars(Object.assign(filterParameters(), {
                        page: page,
                        page_size: 20,
                        sort: document.getElementById("filter-sort").value
                    }));
                renderCards(results, response.items, images);
                updatePager(response.page, response.page_size, response.total_items, response.total_pages);
                toggleState(loading, error, false, "");
                if (!response.items.length) {
                    var empty = document.createElement("div");
                    empty.className = "col-12 alert alert-secondary";
                    empty.textContent = "No cars match these selections.";
                    results.append(empty);
                }
            } catch (requestError) {
                updatePager(1, 20, 0, 0);
                toggleState(loading, error, false, "Could not load cars: " + requestError.message);
            }
        }

        filterForm.addEventListener("submit", function (event) {
            event.preventDefault();
            searchQuery = "";
            searchInput.value = "";
            loadPage(1);
        });
        document.getElementById("car-filter-reset").addEventListener("click", function () {
            window.setTimeout(function () {
                manufacturerSelect.value = "";
                modelSelect.disabled = true;
                addOptions(modelSelect, [], "Choose a manufacturer first");
                loadPage(1);
            }, 0);
        });
        searchForm.addEventListener("submit", function (event) {
            event.preventDefault();
            searchQuery = searchInput.value.trim();
            filterForm.reset();
            manufacturerSelect.value = "";
            modelSelect.disabled = true;
            addOptions(modelSelect, [], "Choose a manufacturer first");
            loadPage(1);
        });

        await loadPage(1);
    }

    async function setupBrandCatalog() {
        var catalog = document.getElementById("brand-api-catalog");
        if (!catalog) return;

        var loading = document.getElementById("brand-api-loading");
        var error = document.getElementById("brand-api-error");
        var results = document.getElementById("brand-api-results");
        var brandName = catalog.dataset.manufacturer;
        var images = collectImages(document.querySelector(".col-md-8") || document);
        var page = 1;
        var updatePager = setupPager({ previous: "brand-page-previous", next: "brand-page-next", label: "brand-page-label" }, loadPage);

        async function loadPage(nextPage) {
            page = nextPage;
            toggleState(loading, error, true, "");
            try {
                var response = await CarWorldAPI.getCars({ manufacturer: brandName, page: page, page_size: 12, sort: "name" });
                renderCards(results, response.items, images);
                updatePager(response.page, response.page_size, response.total_items, response.total_pages);
                toggleState(loading, error, false, "");
            } catch (requestError) {
                toggleState(loading, error, false, "Live listings are unavailable. Showing the saved listings below.");
            }
        }

        await loadPage(1);
    }

    document.addEventListener("DOMContentLoaded", function () {
        setupHomeCatalog();
        setupBrandCatalog();
    });
})();