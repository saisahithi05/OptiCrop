// ==========================================================
// OptiCrop
// Frontend JavaScript
// ==========================================================


// ================================================
// DOM Loaded
// ================================================

document.addEventListener("DOMContentLoaded", () => {

    console.log("OptiCrop Loaded Successfully");

    // ================================================
    // Navbar Shadow
    // ================================================

    const navbar = document.querySelector(".navbar");

    window.addEventListener("scroll", () => {

        if (window.scrollY > 30) {

            navbar.style.boxShadow =
                "0 10px 30px rgba(0,0,0,.15)";

        }

        else {

            navbar.style.boxShadow =
                "0 5px 20px rgba(0,0,0,.08)";

        }

    });


    // ================================================
    // Scroll To Top Button
    // ================================================

    const topBtn = document.getElementById("topBtn");

    window.addEventListener("scroll", () => {

        if (window.scrollY > 500) {

            topBtn.style.display = "block";

        }

        else {

            topBtn.style.display = "none";

        }

    });


    topBtn.addEventListener("click", () => {

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    });


    // ================================================
    // Form Elements
    // ================================================

    const form = document.getElementById("predictionForm");

    const predictBtn = document.querySelector(".predict-btn");


    // ================================================
    // Result Elements
    // ================================================

    const cropName = document.getElementById("cropName");

    const cropDescription =
        document.getElementById("cropDescription");

    const cropImage =
        document.getElementById("cropImage");

    const confidence =
        document.getElementById("confidence");


    // Crop Details

    const season =
        document.getElementById("season");

    const harvest =
        document.getElementById("harvest");

    const soil =
        document.getElementById("soil");

    const water =
        document.getElementById("water");

    const fertilizer =
        document.getElementById("fertilizer");

    const market =
        document.getElementById("market");

    const regions =
        document.getElementById("regions");

    const uses =
        document.getElementById("uses");


    // Growing Conditions

    const temperatureRange =
        document.getElementById("temperatureRange");

    const humidityRange =
        document.getElementById("humidityRange");

    const phRange =
        document.getElementById("phRange");

    const rainfallRange =
        document.getElementById("rainfallRange");


    // Tips

    const tipsList =
        document.getElementById("tipsList");
    // ================================================
    // Prediction Form Submit
    // ================================================

    form.addEventListener("submit", async (e) => {

        e.preventDefault();

        const inputData = {

            N: Number(document.getElementById("N").value),

            P: Number(document.getElementById("P").value),

            K: Number(document.getElementById("K").value),

            temperature: Number(document.getElementById("temperature").value),

            humidity: Number(document.getElementById("humidity").value),

            ph: Number(document.getElementById("ph").value),

            rainfall: Number(document.getElementById("rainfall").value)

        };


        // Loading State

        predictBtn.disabled = true;

        predictBtn.innerHTML =
            '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';


        try{

            const response = await fetch("/predict",{

                method:"POST",

                headers:{

                    "Content-Type":"application/json"

                },

                body:JSON.stringify(inputData)

            });


            const result = await response.json();


            if(result.error){

                throw new Error(result.error);

            }


            // ============================================
            // Show Result
            // ============================================

            document.getElementById("resultSection")
            .scrollIntoView({

                behavior:"smooth"

            });


            cropName.textContent =
                result.recommended_crop;

            cropDescription.textContent =
                result.details.description;

            confidence.textContent =
                result.confidence;

            cropImage.src =
                result.details.image;

            cropImage.alt =
                result.recommended_crop;


            // Crop Information

            season.textContent =
                result.details.season;

            harvest.textContent =
                result.details.harvest_time;

            soil.textContent =
                result.details.soil_type;

            water.textContent =
                result.details.water_requirement;

            fertilizer.textContent =
                result.details.fertilizer;

            market.textContent =
                result.details.market_demand;

            regions.textContent =
                result.details.suitable_regions;

            uses.textContent =
                result.details.uses;
            // ============================================
            // Growing Conditions
            // ============================================

            temperatureRange.textContent =
                result.details.temperature;

            humidityRange.textContent =
                result.details.humidity;

            phRange.textContent =
                result.details.ph;

            rainfallRange.textContent =
                result.details.rainfall;


            // ============================================
            // Growing Tips
            // ============================================

            tipsList.innerHTML = "";


            if (Array.isArray(result.details.growing_tips)) {

                result.details.growing_tips.forEach((tip) => {

                    const li = document.createElement("li");

                    li.textContent = tip;

                    tipsList.appendChild(li);

                });

            }

            else {

                const li = document.createElement("li");

                li.textContent =
                    "No growing tips available.";

                tipsList.appendChild(li);

            }

        }

        catch(error){

            console.error(error);

            alert("Prediction Failed!\n\n" + error.message);

        }

        finally{

            predictBtn.disabled = false;

            predictBtn.innerHTML =

                '<i class="fa-solid fa-seedling"></i> Predict Best Crop';

        }

    });
    // ================================================
    // Fade-in Animation
    // ================================================

    const observer = new IntersectionObserver((entries) => {

        entries.forEach((entry) => {

            if (entry.isIntersecting) {

                entry.target.classList.add("show");

            }

        });

    }, {

        threshold: 0.15

    });


    document.querySelectorAll(

        ".detail-card, .condition-card, .workflow-card, .dataset-card, .tips-card, .result-card"

    ).forEach((element) => {

        observer.observe(element);

    });


    // ================================================
    // Default Image Fallback
    // ================================================

    cropImage.addEventListener("error", () => {

        cropImage.src =
            "/static/images/default.jpg";

    });


    // ================================================
    // Enter Key Support
    // ================================================

    document.querySelectorAll("input").forEach((input) => {

        input.addEventListener("keypress", (e) => {

            if (e.key === "Enter") {

                e.preventDefault();

                form.requestSubmit();

            }

        });

    });


    // ================================================
    // Console Message
    // ================================================

    console.log(

        "%c🌱 OptiCrop Ready!",

        "color:green;font-size:18px;font-weight:bold;"

    );

});

