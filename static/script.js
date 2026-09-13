let currentThreadId =
    localStorage.getItem("travel_thread_id") || null;

let latestAnswerMarkdown = "";


// ============================================================
// QUICK PROMPTS
// ============================================================

function setPrompt(text) {
    document.getElementById("userInput").value = text;
}


// ============================================================
// LOADING STATE
// ============================================================

function setLoading(isLoading) {

    const sendBtn =
        document.getElementById("sendBtn");

    const btnText =
        document.getElementById("btnText");

    const btnLoader =
        document.getElementById("btnLoader");

    sendBtn.disabled = isLoading;

    if (isLoading) {

        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");

    } else {

        btnText.classList.remove("hidden");
        btnLoader.classList.add("hidden");
    }
}


// ============================================================
// APPROVAL LOADING STATE
// ============================================================

function setApprovalLoading(isLoading) {

    const approveBtn =
        document.getElementById("approveBtn");

    const reviseBtn =
        document.getElementById("reviseBtn");

    if (approveBtn) {

        approveBtn.disabled = isLoading;

        if (isLoading) {

            approveBtn.textContent =
                "Processing...";

        } else {

            approveBtn.textContent =
                "Approve & Generate Final";
        }
    }

    if (reviseBtn) {

        reviseBtn.disabled = isLoading;

        if (isLoading) {

            reviseBtn.textContent =
                "Processing...";

        } else {

            reviseBtn.textContent =
                "Revise Using Feedback";
        }
    }
}


// ============================================================
// ERROR HANDLING
// ============================================================

function showError(message) {

    const errorBox =
        document.getElementById("errorBox");

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );
}


function hideError() {

    const errorBox =
        document.getElementById("errorBox");

    errorBox.classList.add(
        "hidden"
    );

    errorBox.textContent = "";
}


// ============================================================
// MARKDOWN RENDERING
// ============================================================

function renderMarkdown(
    element,
    text
) {

    if (!element) {
        return;
    }

    if (
        typeof marked !== "undefined"
    ) {

        element.innerHTML =
            marked.parse(
                text || ""
            );

    } else {

        element.innerText =
            text || "";
    }
}


// ============================================================
// WORKFLOW INFORMATION
// ============================================================

function showWorkflow(data) {

    const workflowSection =
        document.getElementById(
            "workflowSection"
        );

    const guardrailBadge =
        document.getElementById(
            "guardrailBadge"
        );

    const supervisorReasoning =
        document.getElementById(
            "supervisorReasoning"
        );

    const agentChips =
        document.getElementById(
            "agentChips"
        );


    if (!workflowSection) {
        return;
    }


    /*
     * Guardrail status
     */

    if (guardrailBadge) {

        if (
            data.guardrail_allowed === false
        ) {

            guardrailBadge.textContent =
                "Guardrail blocked";

            guardrailBadge.classList.add(
                "blocked"
            );

        } else {

            guardrailBadge.textContent =
                "Guardrail passed";

            guardrailBadge.classList.remove(
                "blocked"
            );
        }
    }


    /*
     * Supervisor reasoning
     */

    if (supervisorReasoning) {

        supervisorReasoning.textContent =
            data.supervisor_reasoning ||
            "";
    }


    /*
     * Agent chips
     */

    if (agentChips) {

        agentChips.innerHTML = "";

        const agents =
            data.selected_agents || [];


        agents.forEach(
            function(agent) {

                const chip =
                    document.createElement(
                        "span"
                    );

                chip.className =
                    "agent-chip";

                chip.textContent =
                    agent.replace(
                        "_",
                        " "
                    );

                agentChips.appendChild(
                    chip
                );
            }
        );
    }


    workflowSection.classList.remove(
        "hidden"
    );
}


// ============================================================
// SHOW / HIDE HUMAN APPROVAL SECTION
// ============================================================

function showApprovalSection(
    approvalRequest
) {

    const approvalSection =
        document.getElementById(
            "approvalSection"
        );

    const approvalRequestElement =
        document.getElementById(
            "approvalRequest"
        );

    const feedbackInput =
        document.getElementById(
            "approvalFeedback"
        );


    if (!approvalSection) {

        console.error(
            "approvalSection was not found in index.html"
        );

        return;
    }


    /*
     * Show the approval message.
     */

    if (approvalRequestElement) {

        approvalRequestElement.textContent =
            approvalRequest ||
            "Please review the draft itinerary before the final plan is generated.";
    }


    /*
     * Clear previous feedback.
     */

    if (feedbackInput) {

        feedbackInput.value = "";
    }


    /*
     * Show the HITL section.
     */

    approvalSection.classList.remove(
        "hidden"
    );


    /*
     * Scroll the user to the approval area.
     */

    setTimeout(
        function() {

            approvalSection.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

        },
        150
    );
}


function hideApprovalSection() {

    const approvalSection =
        document.getElementById(
            "approvalSection"
        );

    if (!approvalSection) {
        return;
    }

    approvalSection.classList.add(
        "hidden"
    );
}


// ============================================================
// SHOW RESULT
// ============================================================

function showResult(
    answer,
    threadId,
    data
) {

    latestAnswerMarkdown =
        answer || "";


    const resultSection =
        document.getElementById(
            "resultSection"
        );

    const resultBox =
        document.getElementById(
            "resultBox"
        );

    const threadInfo =
        document.getElementById(
            "threadInfo"
        );


    /*
     * Render the itinerary.
     */

    renderMarkdown(
        resultBox,
        answer
    );


    /*
     * Display thread ID.
     */

    if (threadInfo) {

        threadInfo.textContent =
            `Thread ID: ${threadId}`;
    }


    /*
     * Show result section.
     */

    if (resultSection) {

        resultSection.classList.remove(
            "hidden"
        );
    }


    /*
     * Show workflow information.
     */

    showWorkflow(data);


    /*
     * VERY IMPORTANT:
     *
     * LangGraph returns requires_approval=true
     * when execution pauses at interrupt().
     *
     * In that case we show the existing
     * approvalSection from index.html.
     */

    if (
        data.requires_approval === true
    ) {

        showApprovalSection(
            data.approval_request
        );

    } else {

        hideApprovalSection();
    }


    /*
     * Scroll to result.
     */

    if (resultSection) {

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }
}


// ============================================================
// SEND TRAVEL REQUEST
// ============================================================

async function sendMessage() {

    hideError();

    hideApprovalSection();


    const input =
        document.getElementById(
            "userInput"
        );

    const message =
        input.value.trim();


    if (!message) {

        showError(
            "Please enter your travel request first."
        );

        return;
    }


    setLoading(true);


    try {

        const response =
            await fetch(
                "/api/travel",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        message:
                            message,

                        thread_id:
                            currentThreadId
                    })
                }
            );


        const data =
            await response.json();


        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error ||
                "Something went wrong."
            );
        }


        /*
         * Save the LangGraph thread.
         */

        currentThreadId =
            data.thread_id;


        localStorage.setItem(
            "travel_thread_id",
            currentThreadId
        );


        /*
         * Debug information.
         *
         * This will help us verify that
         * LangGraph actually returned true.
         */



        /*
         * Display result + HITL.
         */

        showResult(
            data.answer,
            data.thread_id,
            data
        );


    } catch (error) {

        console.error(
            "Travel request error:",
            error
        );

        showError(
            error.message ||
            "Could not generate the travel plan."
        );

    } finally {

        setLoading(false);
    }
}


// ============================================================
// HUMAN APPROVAL / REVISION
// ============================================================

async function submitApproval(
    approved
) {

    hideError();


    if (!currentThreadId) {

        showError(
            "No active travel planning thread was found."
        );

        return;
    }


    const feedbackInput =
        document.getElementById(
            "approvalFeedback"
        );


    const feedback =
        feedbackInput
            ? feedbackInput.value.trim()
            : "";


    /*
     * Revision requires feedback.
     */

    if (
        approved === false &&
        !feedback
    ) {

        showError(
            "Please provide revision feedback before requesting a revision."
        );

        if (feedbackInput) {

            feedbackInput.focus();
        }

        return;
    }


    setApprovalLoading(true);


    try {

        const response =
            await fetch(
                "/api/travel/approve",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        thread_id:
                            currentThreadId,

                        approved:
                            approved,

                        feedback:
                            feedback
                    })
                }
            );


        const data =
            await response.json();


        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error ||
                "Could not process the approval."
            );
        }


        /*
         * The SAME LangGraph thread
         * is being resumed.
         */

        currentThreadId =
            data.thread_id;


        localStorage.setItem(
            "travel_thread_id",
            currentThreadId
        );


        /*
         * Update final/draft result.
         */

        latestAnswerMarkdown =
            data.answer || "";


        const resultBox =
            document.getElementById(
                "resultBox"
            );


        renderMarkdown(
            resultBox,
            data.answer || ""
        );


        /*
         * Update thread ID.
         */

        const threadInfo =
            document.getElementById(
                "threadInfo"
            );


        if (threadInfo) {

            threadInfo.textContent =
                `Thread ID: ${currentThreadId}`;
        }


        /*
         * Update workflow information.
         */

        showWorkflow(data);


        /*
         * If another interrupt occurs,
         * show approval again.
         */

        if (
            data.requires_approval === true
        ) {

            showApprovalSection(
                data.approval_request
            );

        } else {

            /*
             * HITL completed.
             */

            hideApprovalSection();


            /*
             * Show completion message.
             */

            showCompletionMessage(
                approved
                    ? "✓ Itinerary approved and final plan generated."
                    : "✓ Itinerary revised using your feedback and final plan generated."
            );
        }


        /*
         * Scroll back to the final result.
         */

        const resultSection =
            document.getElementById(
                "resultSection"
            );


        if (resultSection) {

            resultSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }


    } catch (error) {

        console.error(
            "Approval error:",
            error
        );


        showError(
            error.message ||
            "Could not process the approval."
        );

    } finally {

        setApprovalLoading(false);
    }
}


// ============================================================
// COMPLETION MESSAGE
// ============================================================

function showCompletionMessage(
    message
) {

    const resultSection =
        document.getElementById(
            "resultSection"
        );


    if (!resultSection) {
        return;
    }


    const existing =
        document.getElementById(
            "completionMessage"
        );


    if (existing) {

        existing.remove();
    }


    const completion =
        document.createElement(
            "div"
        );


    completion.id =
        "completionMessage";


    completion.className =
        "completion-message";


    completion.textContent =
        message;


    const resultBox =
        document.getElementById(
            "resultBox"
        );


    if (resultBox) {

        resultBox.insertAdjacentElement(
            "afterend",
            completion
        );

    } else {

        resultSection.appendChild(
            completion
        );
    }
}


// ============================================================
// COPY RESULT
// ============================================================

function copyResult() {

    const resultBox =
        document.getElementById(
            "resultBox"
        );


    if (!resultBox) {
        return;
    }


    const text =
        resultBox.innerText;


    if (!text) {
        return;
    }


    navigator.clipboard
        .writeText(text)

        .then(() => {

            const copyBtn =
                document.querySelector(
                    ".copy-btn"
                );


            if (!copyBtn) {
                return;
            }


            const oldText =
                copyBtn.textContent;


            copyBtn.textContent =
                "Copied!";


            setTimeout(
                function() {

                    copyBtn.textContent =
                        oldText;

                },
                1400
            );
        })

        .catch(() => {

            showError(
                "Could not copy result."
            );
        });
}


// ============================================================
// DOWNLOAD PDF
// ============================================================

function downloadPDF() {

    const pdfContent =
        document.getElementById(
            "pdfContent"
        );


    if (
        !latestAnswerMarkdown ||
        !pdfContent
    ) {

        showError(
            "No travel plan available to download."
        );

        return;
    }


    const downloadBtn =
        document.querySelector(
            ".download-btn"
        );


    if (!downloadBtn) {
        return;
    }


    const oldText =
        downloadBtn.textContent;


    downloadBtn.textContent =
        "Preparing PDF...";


    downloadBtn.disabled =
        true;


    const options = {

        margin: 0.5,

        filename:
            "ai-travel-plan.pdf",

        image: {

            type: "jpeg",

            quality: 0.98
        },

        html2canvas: {

            scale: 2,

            useCORS: true,

            backgroundColor:
                "#ffffff"
        },

        jsPDF: {

            unit: "in",

            format: "a4",

            orientation:
                "portrait"
        },

        pagebreak: {

            mode: [
                "avoid-all",
                "css",
                "legacy"
            ]
        }
    };


    html2pdf()

        .set(options)

        .from(pdfContent)

        .save()

        .then(() => {

            downloadBtn.textContent =
                oldText;

            downloadBtn.disabled =
                false;
        })

        .catch(() => {

            downloadBtn.textContent =
                oldText;

            downloadBtn.disabled =
                false;

            showError(
                "Could not download PDF."
            );
        });
}


// ============================================================
// KEYBOARD SHORTCUT
// ============================================================

document.addEventListener(
    "keydown",
    function(event) {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            sendMessage();
        }
    }
);