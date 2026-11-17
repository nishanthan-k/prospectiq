from app.research.ranker import rank_chunks, score_chunk


def test_score_chunk_prioritizes_criteria_matches():
    chunk_relevant = (
        "Acme provides a B2B SaaS platform for automated SOC2 compliance and developer APIs."
    )
    chunk_irrelevant = (
        "Please read our cookie policy and privacy policy before signing in to your account."
    )

    criteria = ["B2B SaaS", "SOC2 compliance"]
    score_rel, matched_rel = score_chunk(
        chunk=chunk_relevant, company_name="Acme", qualification_criteria=criteria
    )
    score_irrel, matched_irrel = score_chunk(
        chunk=chunk_irrelevant, company_name="Acme", qualification_criteria=criteria
    )

    assert score_rel > score_irrel
    assert "B2B SaaS" in matched_rel or "SOC2 compliance" in matched_rel
    assert len(matched_irrel) == 0


def test_score_chunk_with_context_summary():
    chunk = "Retool is a low-code platform for building internal developer tools."
    score_with_ctx, _ = score_chunk(
        chunk=chunk, company_name="Retool", context_summary="internal tooling for developers"
    )
    score_without_ctx, _ = score_chunk(chunk=chunk, company_name="Retool")
    assert score_with_ctx > score_without_ctx


def test_score_chunk_penalizes_recruitment_postings():
    job_chunk = (
        "Job Opening 01 Associate Attorney We're hiring litigation lawyers! "
        "Looking for talents with experience in court filings, employment type full time."
    )
    service_chunk = (
        "We provide comprehensive commercial litigation and intellectual property legal services "
        "tailored to enterprise clients."
    )
    score_job, _ = score_chunk(
        job_chunk, "Sterling Law", context_summary="commercial litigation legal services"
    )
    score_service, _ = score_chunk(
        service_chunk, "Sterling Law", context_summary="commercial litigation legal services"
    )
    assert score_service > score_job


def test_rank_chunks_orders_and_limits_results():
    chunk_tuples = [
        ("https://example.com/legal", "Terms of service and privacy policy for users."),
        (
            "https://example.com/product",
            "Acme Enterprise platform delivers automated developer APIs and multi-cloud scaling.",
        ),
        (
            "https://example.com/pricing",
            "Acme pricing tiers include Pro and Enterprise with dedicated customer support.",
        ),
    ]

    ranked = rank_chunks(
        chunk_tuples,
        company_name="Acme",
        qualification_criteria=["developer APIs", "Enterprise tier"],
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].url in ["https://example.com/product", "https://example.com/pricing"]
    assert ranked[0].score >= ranked[1].score
