#pragma once


#include <string>

#include <network/network.hpp>
#include <stratum/stratum.hpp>


namespace stratum
{
    // Monero-family stratum (RandomX / rx/0), as spoken by XMRig-compatible pools:
    // a JSON-RPC `login` handshake returns the session id and the first job; further
    // jobs arrive via the unsolicited `job` method; shares are sent with `submit`
    // (job_id + nonce + result hash).
    class StratumMonero : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void onUnknownMethod(boost::json::object const& root) final;

        // onConnect() (base) dispatches to miningSubscribe() for the default stratum type;
        // we repurpose it to send the Monero `login` request.
        void miningSubscribe() final;
        void miningSubmit(uint32_t const deviceId, boost::json::object const& params) final;

      private:
        // RPC session id returned by `login`, echoed back in every `submit`.
        std::string rpcId{};

        void parseJob(boost::json::object const& job);
    };
}
